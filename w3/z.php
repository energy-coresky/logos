<?php

require 'gpt_pack.php';
require 'param.php';

//var_export($state_dict);
#file_put_contents('gpt-model-vINT5.bin', GPT_Pack::encode($state_dict, GPT_Pack::Q_INT4));die;

//$state_dict = GPT_pack::decode(file_get_contents('gpt-model-vINT4.bin'));
#$state_dict = GPT_pack::decode(file_get_contents('gpt-model-vFP32.bin'));

#var_export($state_dict);


/*
*/

// Настройки генерации
$filename = 'math.txt';      // Имя выходного файла
$total_samples = 2000;       // Сколько примеров сгенерировать (с запасом для обучения)
$max_val = 9;               // Максимальное число для слагаемых
$block_size = 8;             // Твоё ограничение контекста

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
