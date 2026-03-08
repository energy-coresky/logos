<?php

namespace logos;
use Plan, GPT_Run, GPT_Bin, GPT_Data;

class app extends \Console
{
    function __construct($argv = [], $found = []) {
        Plan::set('logos', fn() => parent::__construct($argv, $found));
    }

    function data($in, $cfg) {
        $v = (object)($cfg->v + $cfg->default);
        $v->settings = $this->args($in, $cfg->default, function (&$key, $val) use ($v, $cfg) {
            if ($set = isset($cfg->short[$key]))
                $key = $cfg->short[$key];
            $v->$key = $val;
            return !$set;
        });
        return $v;
    }

    /** Run train &| inference */
    function a_z(...$in) {
        $v = $this->data($in, $cfg = cfg('gpt'));
        if (!$v->txt && !$v->bin)
            return $this->a_u();
        $time = time();
        if ($load_bin = !$v->qtz && $v->bin) {
            [$v->settings, $state_dict] = GPT_Bin::load($v->bin);
            $v->txt = $v->settings['dataset'];
        }
        $v->rnd && mt_srand($v->rnd);
        $gpt = new GPT_Run($v->settings);
        if ($load_bin)
            $gpt->params = $state_dict;
        $docs = $gpt->build_vocab($v->txt);
        
        echo "num docs: " . count($docs) . "\n";
        echo "vocab size: $gpt->vocab_size\n";
        echo "num params: $gpt->n_params\n=============\n";
        foreach ($v->settings as $key => $val)
            echo "$key: $val\n";

        if (!$load_bin) {
            echo "TRAIN:\n";
            foreach ($gpt->train($docs, $v->n_train, $v->learning_rate) as $i => $loss)
                echo "\r  step $i | loss $loss | seconds " . (time() - $time) . '     ';
            if ($v->qtz) {
                $qtz = $v->qtz == 4 ? GPT_Bin::Q_INT4 : ($v->qtz == 8 ? GPT_Bin::Q_INT8 : GPT_Bin::Q_FP32);
                $v->settings['dataset'] = $v->txt;
                GPT_Bin::save($v->bin, $gpt->params, $v->settings, $qtz);
            }
            echo "\n";
        }
        echo "INFERENCE:";
        foreach ($gpt->inference($v->temperature, $v->n_inference) as $i => $sample)
            echo "\n  sample $i: $sample";
    }

    /** Run bin-file inference as chat */
    function a_chat(...$in) {
        $v = $this->data($in, $cfg = cfg('gpt'));
        if (!$v->bin)
            return $this->a_u();
        [$v->settings, $params] = GPT_Bin::load($v->bin);
        $v->rnd && mt_srand($v->rnd);
        $gpt = new GPT_Run($v->settings, $params);
        $docs = $gpt->build_vocab($v->settings['dataset']);
        
        echo "num params: $gpt->n_params\n=============\n";
        foreach ($v->settings as $key => $val)
            echo "$key: $val\n";

        shuffle($docs);
        echo "INFERENCE:";
        for (;1;) {
            $p = trim(fgets(STDIN));
            foreach ($gpt->inference($v->temperature, 3, $p) as $response)
                echo " response: $p$response\n";
        }
    }

    /** Generate the datasets */
    function a_g(...$in) { // names datasets - https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt
        $v = $this->data($in, $cfg = cfg('data'));
        if (method_exists($gen = new GPT_Data, $type = $v->type)) {
            $gen->$type($v->settings);
        } else echo strtr($cfg->usage, [
            '%list_1%' => var_export($cfg->short, true),
            '%list_2%' => var_export($v->settings, true),
        ]);
    }

    /** See Logos usage (how run the model) */
    function a_u() {
        $cfg = cfg('gpt');
        echo strtr($cfg->usage, [
            '%list_1%' => var_export($cfg->short, true),
            '%list_2%' => var_export($cfg->default, true),
        ]);
    }
}
