<?php

class GPT_Data
{
    function math() {

        $filename = 'txt/math.txt';
        $total_samples = 2000;
        $max_val = 9;
        $block_size = 8;
        $dataset = [];
        $generated = 0;

        echo "Генерация датасета '$filename' (макс длина строки: $block_size)...\n";
        // Используем while, чтобы набрать нужное количество примеров
        while ($generated < $total_samples) {
            // Генерируем случайные слагаемые
            $a = rand(0, $max_val);
            $b = rand(0, $max_val);
            $sum = $a + $b;
            $line = "{$a}+{$b}={$sum}";
            if (strlen($line) < $block_size) {
                $dataset[] = $line . "\n";
                $generated++;
            }
        }

        // Перемешиваем массив, чтобы модели было сложнее "запомнить" порядок
        shuffle($dataset);
        // Сохраняем в файл
        file_put_contents($filename, implode('', $dataset));

        echo "Готово! Сгенерировано $generated примеров.\n";
        echo "Примеры строк:\n";
        echo implode('', array_slice($dataset, 0, 5));
    }
}
