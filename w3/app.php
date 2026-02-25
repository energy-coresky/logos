<?php

namespace logos;
use Plan;

class app extends \Console
{
    function __construct($argv = [], $found = []) {
        Plan::set('logos', fn() => parent::__construct($argv, $found));
    }

    /** Run train & inference */
    function a_run() {
        //var_export($state_dict);
        require 'param.php';

        #file_put_contents('gpt-model-vINT5.bin', GPT_Pack::encode($state_dict, GPT_Pack::Q_INT4));die;

        //$state_dict = GPT_pack::decode(file_get_contents('gpt-model-vINT4.bin'));
        #$state_dict = GPT_pack::decode(file_get_contents('gpt-model-vFP32.bin'));

        #var_export($state_dict);
        print(11);
    }

    /** Generate the dataset */
    function a_gen() {
        print(11);
    }
}
