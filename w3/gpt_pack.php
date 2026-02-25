<?php

class GPT_Pack
{
    // Константы типов квантования
    const Q_FP32 = 0; // 4 байта на вес (стандартный float)
    const Q_INT8 = 1; // 1 байт на вес (сжатие x4)
    const Q_INT4 = 2; // 0.5 байта на вес (сжатие x8)

    /**
     * Упаковка state_dict в бинарную строку
     *
     * @param array $state_dict Трехмерный массив весов ['layer_name' => [...weights...]]
     * @param int $quantization Тип квантования (self::Q_INT8, etc.)
     * @return string Бинарная строка для записи в файл
     */
    public static function encode(array $state_dict, int $quantization = self::Q_INT8): string
    {
        // 1. Формируем заголовок файла (12 байт)
        // Magic (4 байта) + Version (1 байт) + Quant Type (1 байт) + Layers Count (4 байта) + Reserved (2 байта)
        $header = pack('a4CCVCC', 
            'GPT1',        // Magic string (идентификатор формата)
            1,             // Version (версия формата)
            $quantization, // Тип квантования
            count($state_dict), // Количество слоев
            1,1              // Reserved
        );
        // 2. Формируем индекс (таблицу метаданных слоев)
        // Каждая запись: NameLen(2) + Name(N) + DimsCount(1) + Dims(4*N) + DataLen(4)
        // Временный буфер для самих данных, чтобы сначала записать индекс, потом данные
        $dataBlob = '';

        foreach ($state_dict as $layerName => $weights) {
            // Сплющиваем массив весов в одномерный вектор
            $flatWeights = self::flatten($weights);
            // Запоминаем размерности, чтобы при декодировании восстановить форму
            $shape = self::getShape($weights);
            // Pack: Short(NameLen) + String(Name) + Char(DimsCount) + Long*Dims + Long(DataLen)
            $header .= pack('S', strlen($layerName)) . $layerName;
            $header .= pack('C', count($shape)); // Записываем метаданные слоя в индекс
            foreach ($shape as $dim)
                $header .= pack('L', $dim); // Long (4 байта) на размерность
            // Квантуем данные и получаем бинарную строку
            $header .= pack('L', strlen($packedData = self::quantize($flatWeights, $quantization)));
            $dataBlob .= $packedData;
        }
        return $header . $dataBlob;
    }

    /**
     * Распаковка бинарной строки обратно в state_dict
     *
     * @param string $binary Бинарные данные из файла
     * @return array|null Восстановленный массив весов или null при ошибке
     */
    public static function decode(string $binary): ?array
    {
        // "C1char1/C1char2" - такой синтаксис позволяет именовать ключи
        $headerFormat = 'a4magic/Cversion/Cquantization/Vlayer_count/Cres1/Cres2';
        // unpack не меняет указатель, нам нужно самим считать длину формата
        $offset = 12; 
        $header = unpack($headerFormat, $binary, 0);

        if ($header['magic'] !== 'GPT1') {
            return null; // Не тот формат
        }

        $quantization = $header['quantization'];
        $layerCount = $header['layer_count'];
        $state_dict = [];

        // 2. Читаем индекс
        for ($i = 0; $i < $layerCount; $i++) {
            // Читаем длину имени (Short - 2 байта)
            $meta = unpack('Sname_len', $binary, $offset);
            $offset += 2;

            // Читаем имя слоя
            $layerName = substr($binary, $offset, $meta['name_len']);
            $offset += $meta['name_len'];

            // Читаем количество размерностей (Char - 1 байт)
            $meta = unpack('Cdims_count', $binary, $offset);
            $offset += 1;
            $dimsCount = $meta['dims_count'];

            // Читаем сами размерности (Long - 4 байта каждая)
            $shape = [];
            $shapeFormat = '';
            for ($d = 0; $d < $dimsCount; $d++)
                $shapeFormat .= ($d ? '/' : '') . 'Ldim' . $d;
            
            $shapeData = unpack($shapeFormat, $binary, $offset);
            $offset += 4 * $dimsCount;
            
            // unpack возвращает массив ['dim0'=>val, 'dim1'=>val], нам нужны значения
            $shape = array_values($shapeData);
            // Читаем длину блока данных (Long - 4 байта)
            $meta = unpack('Ldata_len', $binary, $offset);
            $offset += 4;
            $state_dict[$layerName] = [$shape, $meta['data_len']];
        }

        // Деквантуем и восстанавливаем форму
        foreach ($state_dict as &$v) {
            [$shape, $dataLen] = $v;
            // Вырезаем бинарный блок с весами
            $v = self::dequantize(substr($binary, $offset, $dataLen), $shape, $quantization);
            $offset += $dataLen;
        }

        return $state_dict;
    }

    /**
     * Квантование одномерного массива весов
     */
    private static function quantize(array $weights, int $type): string
    {
        $count = count($weights);
        $binary = '';

        if ($type === self::Q_FP32) {
            // Просто упаковываем float'ы (little-endian)
            foreach ($weights as $w) {
                $binary .= pack('g', (float)$w);
            }
        } 
        elseif ($type === self::Q_INT8) {
            // Находим максимальное абсолютное значение в слое для масштабирования
            $maxAbs = 0.0;
            foreach ($weights as $w) {
                $val = abs($w);
                if ($val > $maxAbs) $maxAbs = $val;
            }
            $scale = $maxAbs / 127.0;

            // Сохраняем Scale (float) в начале блока данных, чтобы при декодировании восстановить
            $binary .= pack('g', $scale);

            // Упаковываем веса
            foreach ($weights as $w) {
                // Приводим к диапазону -127..127
                $val = round($w / $scale);
                $val = max(-128, min(127, $val)); // Clip
                $binary .= pack('c', (int)$val); // signed char
            }
        } 
        elseif ($type === self::Q_INT4) {
            // === АСИММЕТРИЧНАЯ INT4 (0..15) ===
            // Вычисляем масштаб и сдвиг (Zero Point)
            // Формула: val_quant = round( (val - min) / (max - min) * 15 )
            // Диапазон: 0..15

            // Сохраняем Min и Max (по 4 байта каждый) для точного восстановления
            $binary = pack('g', $minVal = min($weights));
            $binary .= pack('g', $maxVal = max($weights));
            
            $count = count($weights);
            // Паддинг до четного
            if ($count % 2 !== 0) {
                $weights[] = $minVal; // Заполняем значением, которое превратится в 0
                $count++;
            }

            for ($i = 0; $i < $count; $i += 2) {
                // Квантуем первое число в диапазон 0..15
                $v1_norm = ($weights[$i] - $minVal) / ($maxVal - $minVal);
                $v1_int = (int)round($v1_norm * 15);
                $v1_int = max(0, min(15, $v1_int)); // Clip
              
                // Квантуем второе число
                $v2_norm = ($weights[$i+1] - $minVal) / ($maxVal - $minVal);
                $v2_int = (int)round($v2_norm * 15);
                $v2_int = max(0, min(15, $v2_int));
              
                // Пакуем два 4-битных числа в байт
                $byte = ($v1_int << 4) | $v2_int;
                $binary .= pack('C', $byte);
            }
        } 
        elseif ($type === self::Q_INT4__) {
            // Для INT4 нам нужно упаковать 2 числа в 1 байт.
            // Диапазон INT4: -8..7.
            $maxAbs = 0.0;
            foreach ($weights as $w) {
                $val = abs($w);
                if ($val > $maxAbs) $maxAbs = $val;
            }
            // Делим на 7 (максимальное положительное значение в int4)
            $scale = $maxAbs / 7.0;
            
            $binary .= pack('g', $scale);

            // Если нечетное количество, добавляем паддинг
            if ($count % 2 !== 0) {
                $weights[] = 0;
                $count++;
            }

            for ($i = 0; $i < $count; $i += 2) {
                $v1 = (int)round($weights[$i] / $scale);
                $v2 = (int)round($weights[$i+1] / $scale);
                
                // Клиппинг в диапазон -8..7
                $v1 = max(-8, min(7, $v1));
                $v2 = max(-8, min(7, $v2));

                // Сдвигаем в беззнаковый диапазон 0..15 для упаковки
                // -8 -> 0, 0 -> 8, 7 -> 15
                $v1 = $v1 + 8;
                $v2 = $v2 + 8;

                // Упаковываем два 4-битных числа в один байт
                // Первое число в старших битах, второе в младших
                $byte = ($v1 << 4) | $v2;
                $binary .= pack('C', $byte);
            }
        }

        return $binary;
    }

    // --- Вспомогательные функции ---

    /** Рекурсивное преобразование многомерного массива в плоский */
    private static function flatten(array $array): array
    {
        $flat = [];
        foreach (new RecursiveIteratorIterator(new RecursiveArrayIterator($array)) as $value) {
            $flat[] = $value;
        }
        return $flat;
    }

    /** Получение размерностей массива (Shape) */
    private static function getShape(array $ref): array
    {
        $shape = [];
        while (is_array($ref)) {
            $shape[] = count($ref);
            $ref = reset($ref);
        }
        return $shape;
    }

    /**
     * Деквантование и восстановление формы массива
     */
    private static function dequantize(string $blob, array $shape, int $type): array
    {
        // Вычисляем ожидаемое количество элементов
        $expectedCount = array_product($shape);
        $weights = [];

        if ($type === self::Q_FP32) {
            // 'f' или 'g' (little endian float) - 4 байта
            // unpack возвращает массив с индексами 1, 2, 3...
            // Чтобы быстро распаковать много чисел, можно использовать формат "f*"
            // но unpack с * дает ключи 1, 2, 3... что нам и нужно
            $parsed = unpack('g*', $blob);
            $weights = array_values($parsed); // Сбрасываем ключи 1..N в 0..N-1
        } 
        elseif ($type === self::Q_INT8) {
            // Первые 4 байта - Scale (float)
            $scaleData = unpack('gscale', $blob);
            $scale = $scaleData['scale'];
            
            // Остальное - signed char (c*)
            $intWeights = unpack('c*', substr($blob, 4));
            
            foreach ($intWeights as $val) {
                $weights[] = (float)($val * $scale);
            }
        } 
        elseif ($type === self::Q_INT4) {
            // Читаем Min и Max (первые 8 байт)
            if (strlen($blob) < 8) return [];
            
            $params = unpack('gmin_val/gmax_val', $blob);
            $minVal = $params['min_val'];
            $maxVal = $params['max_val'];
            
            $range = $maxVal - $minVal;
            
            $bytes = unpack('C*', substr($blob, 8)); // Данные начинаются с 9-го байта
            $weights = [];
            
            foreach ($bytes as $byte) {
                // Распаковка симметричная
                $v1_int = ($byte >> 4) & 0x0F; // 0..15
                $v2_int = $byte & 0x0F;        // 0..15
                
                // Восстановление значения: val = min + (index / 15) * range
                $w1 = $minVal + ($v1_int / 15.0) * $range;
                $w2 = $minVal + ($v2_int / 15.0) * $range;
                
                $weights[] = $w1;
                $weights[] = $w2;
            }
            
            // Удаление паддинга
            $expectedCount = array_product($shape);
            if (count($weights) > $expectedCount) {
                $weights = array_slice($weights, 0, $expectedCount);
            }
            
            return self::reshape($weights, $shape);
        } 
        elseif ($type === self::Q_INT4__) {
            // Первые 4 байта - Scale
            $scaleData = unpack('gscale', $blob);
            $scale = $scaleData['scale'];
            
            // Остальное - байты, в каждом по 2 веса
            $bytes = unpack('C*', substr($blob, 4));
            
            foreach ($bytes as $byte) {
                // Восстанавливаем два числа из одного байта
                
                // Первое число - старшие 4 бита
                $v1_raw = ($byte >> 4) & 0x0F;
                // Второе число - младшие 4 бита
                $v2_raw = $byte & 0x0F;

                // Обратное преобразование из диапазона 0..15 в -8..7
                $v1 = $v1_raw - 8;
                $v2 = $v2_raw - 8;

                $weights[] = (float)($v1 * $scale);
                $weights[] = (float)($v2 * $scale);
            }
            
            // INT4 упаковывает по 2, поэтому если кол-во весов было нечетным,
            // при упаковке добавлялся паддинг. Убираем лишний элемент, если он есть.
            if (count($weights) > $expectedCount) {
                array_pop($weights);
            }
        }

        // Восстанавливаем многомерную структуру массива
        return self::reshape($weights, $shape);
    }

    /**
     * Преобразование плоского массива в многомерный по форме
     */
    private static function reshape(array $flat, array $shape): array
    {
        $count = count($flat);
        if (empty($shape)) return $flat;
        
        $dim = array_shift($shape);
        $chunkSize = $shape ? (int)array_product($shape) : 1;
        
        $result = [];
        for ($i = 0; $i < $dim; $i++) {
            $slice = array_slice($flat, $i * $chunkSize, $chunkSize);
            if (!empty($shape)) {
                $result[] = self::reshape($slice, $shape);
            } else {
                $result[] = $slice[0] ?? 0; // Последняя размерность - просто число
            }
        }
        
        return $result;
    }
}


