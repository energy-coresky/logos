<?php

class GPT_Data /* datasets generation */
{
    function math(array $settings) {
        extract($settings);
        $dataset = [];
        $n = 0;
        while ($n < $n_samples) {
            $a = rand(0, $max_val);
            $b = rand(0, $max_val);
            $sum = $a + $b;
            $line = "{$a}+{$b}={$sum}";
            
            if (strlen($line) < $block_size) {
                $dataset[] = $line . "\n";
                $n++;
            }
        }
        
        $this->_save($filename, $dataset);
    }

    function dna(array $settings) {
        extract($settings);
        $bases = ['A', 'C', 'G', 'T'];
        $dataset = [];
        for ($i = 0; $i < $n_samples; $i++) {
            // Генерируем случайную длину, влезающую в block_size
            $len = rand(4, max(4, $block_size - 2)); 
            $seq = '';
            for ($j = 0; $j < $len; $j++) {
                $seq .= $bases[array_rand($bases)];
            }
            $dataset[] = $seq . "\n";
            
            // Добавляем паттерны (повторы) для интереса
            if ($i % 5 == 0 && $i > 0) {
                $base1 = $bases[array_rand($bases)];
                $base2 = $bases[array_rand($bases)];
                $pattern = str_repeat($base1 . $base2, ceil($block_size / 2));
                $dataset[] = substr($pattern, 0, $block_size - 1) . "\n";
                $i++;
            }
        }
        
        $this->_save($filename, $dataset);
    }

    protected function _save(string $filename, array $dataset) {
        shuffle($dataset);
        file_put_contents($filename, implode('', $dataset));
        echo "Готово! Сгенерировано " . count($dataset) . " примеров.\n";
        echo "Примеры строк:\n";
        echo implode('', array_slice($dataset, 0, 5));
    }
}
