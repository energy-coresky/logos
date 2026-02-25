<?php

namespace logos;
use Plan, GPT_Run;

class app extends \Console
{
    function __construct($argv = [], $found = []) {
        Plan::set('logos', fn() => parent::__construct($argv, $found));
    }

    /** Run train & inference */
    function a_run($n_steps = 300) { // https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt
        // Let there be order among chaos
        mt_srand(37);
        $lines = array_map('trim', file(self::$d[1] . '/txt/names.txt'));
      //$lines = array_map('trim', file('txt/math.txt'));
        $docs = array_values(array_filter($lines, fn($l) => !empty($l)));
        shuffle($docs);
        echo "num docs: " . count($docs) . "\n";
        $gpt = new GPT_Run($docs);
        echo "vocab size: $gpt->vocab_size\n";
        $gpt->train($docs, $n_steps);
        foreach ($gpt->inference(0.6, 11) as $i => $sample)
            echo "sample $i: " . $sample . "\n";
    }

    /** Generate the dataset */
    function a_gen() {
/* 
        //var_export($state_dict);
        require 'param.php';

        #file_put_contents('gpt-model-vINT5.bin', GPT_Pack::encode($state_dict, GPT_Pack::Q_INT4));die;

        //$state_dict = GPT_pack::decode(file_get_contents('gpt-model-vINT4.bin'));
        #$state_dict = GPT_pack::decode(file_get_contents('gpt-model-vFP32.bin'));

        #var_export($state_dict);
        print(11);
*/
        print(11);
    }
}
