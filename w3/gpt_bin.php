<?php

class GPT_Bin
{
    const Q_FP32 = 1; // 4 байта на вес
    const Q_INT8 = 2; // 1 байт на вес
    const Q_INT4 = 3; // 0.5 байта на вес

    public static function save($name, $state_dict, $settings, $qtz): void
    {
        $len = strlen($json = json_encode($settings));
        $header = pack('a4CV', 'GPT1', 2, $len);
        $header .= $json . pack('CV', (int)$qtz, count($state_dict));
        $blob = '';
        foreach ($state_dict as $layerName => $weights) {
            $flatWeights = self::flatten($weights);
            $shape = self::getShape($weights);
            $header .= pack('S', strlen($layerName)) . $layerName;
            $header .= pack('C', count($shape));
            foreach ($shape as $dim)
                $header .= pack('L', $dim);
            $packed = self::quantize($flatWeights, (int)$qtz);
            $header .= pack('L', strlen($packed));
            $blob .= $packed;
        }
        Plan::bin_p("$name.bin", $header . $blob);
    }

    public static function load(string $name): ?array
    {
        $header = unpack('a4magic/Cversion/Vmeta_len', $bin = Plan::bin_g("$name.bin"), 0);
        if ($header['magic'] !== 'GPT1')
            throw new Error('Error in file format');
        $json = substr($bin, 9, $header['meta_len']);
        $offset = 9 + $header['meta_len'];
        $header = unpack('Cquantization/Vlayer_count', $bin, $offset);
        $offset += 5;
        for ($i = 0; $i < $header['layer_count']; $i++) {
            $meta = unpack('Sname_len', $bin, $offset);
            $offset += 2;
            $layerName = substr($bin, $offset, $meta['name_len']);
            $offset += $meta['name_len'];
            $meta = unpack('Cdims_count', $bin, $offset);
            $offset += 1;
            for ($format = '', $d = 0; $d < $meta['dims_count']; $d++)
                $format .= ($d ? '/' : '') . 'Ldim' . $d;
            $shape = unpack($format, $bin, $offset);
            $offset += 4 * $meta['dims_count'];
            $meta = unpack('Ldata_len', $bin, $offset);
            $offset += 4;
            $state_dict[$layerName] = [array_values($shape), $meta['data_len']];
        }
        foreach ($state_dict as &$v) {
            [$shape, $len] = $v;
            $v = self::dequantize(substr($bin, $offset, $len), $shape, $header['quantization']);
            $offset += $len;
        }
        return [json_decode($json, true), $state_dict];
    }

    private static function quantize(array $weights, int $type): string
    {
        $count = count($weights);
        $bin = '';

        if ($type === self::Q_FP32) {
            foreach ($weights as $w)
                $bin .= pack('g', (float)$w);
        } 
        elseif ($type === self::Q_INT8) {
            $maxAbs = 0.0;
            foreach ($weights as $w)
                if (abs($w) > $maxAbs)
                    $maxAbs = abs($w);
            $scale = $maxAbs / 127.0;
            $bin = pack('g', $scale);

            foreach ($weights as $w) {
                $val = round($w / $scale);
                $bin .= pack('c', (int)max(-128, min(127, $val)));
            }
        }
        elseif ($type === self::Q_INT4) {
            $bin = pack('g', $minVal = min($weights)) . pack('g', $maxVal = max($weights));
            $range = $maxVal - $minVal or $range = 1.0;
            if ($count % 2 !== 0) {
                $weights[] = $minVal;
                $count++;
            }
            for ($i = 0; $i < $count; $i += 2) {
                $v1_int = (int)round((($weights[$i] - $minVal) / $range) * 15);
                $v2_int = (int)round((($weights[$i + 1] - $minVal) / $range) * 15);
                $v1_int = max(0, min(15, $v1_int));
                $v2_int = max(0, min(15, $v2_int));
                $bin .= pack('C', ($v1_int << 4) | $v2_int);
            }
        }
        return $bin;
    }

    private static function dequantize(string $blob, array $shape, int $type): array
    {
        $expectedCount = array_product($shape);
        $weights = [];

        if ($type === self::Q_FP32) {
            $parsed = unpack('g*', $blob);
            $weights = array_values($parsed);
        }
        elseif ($type === self::Q_INT8) {
            $scaleData = unpack('gscale', $blob);
            $scale = $scaleData['scale'];
            $intWeights = unpack('c*', substr($blob, 4));
            foreach ($intWeights as $val)
                $weights[] = (float)($val * $scale);
        }
        elseif ($type === self::Q_INT4) {
            if (strlen($blob) < 8)
                return [];
            $params = unpack('gmin_val/gmax_val', $blob);
            $range = $params['max_val'] - ($minVal = $params['min_val']);
            $bytes = unpack('C*', substr($blob, 8));
            foreach ($bytes as $byte) {
                $v1_int = ($byte >> 4) & 0x0F;
                $v2_int = $byte & 0x0F;
                $weights[] = $minVal + ($v1_int / 15.0) * $range;
                $weights[] = $minVal + ($v2_int / 15.0) * $range;
            }
        }

        if (count($weights) > $expectedCount) {
            $weights = array_slice($weights, 0, $expectedCount);
        }
        return self::reshape($weights, $shape);
    }

    private static function flatten(array $array): array {
        $flat = [];
        foreach (new RecursiveIteratorIterator(new RecursiveArrayIterator($array)) as $value)
            $flat[] = $value;
        return $flat;
    }

    private static function getShape(array $ref): array {
        $shape = [];
        while (is_array($ref)) {
            $shape[] = count($ref);
            $ref = reset($ref);
        }
        return $shape;
    }

    private static function reshape(array $flat, array $shape): array {
        if (empty($shape))
            return $flat;
        $dim = array_shift($shape);
        $chunkSize = $shape ? (int)array_product($shape) : 1;
        $result = [];
        for ($i = 0; $i < $dim; $i++) {
            $slice = array_slice($flat, $i * $chunkSize, $chunkSize);
            $result[] = !empty($shape) ? self::reshape($slice, $shape) : ($slice[0] ?? 0);
        }
        return $result;
    }
}
