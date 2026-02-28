<?php

namespace logos;
use Plan, GPT_Run, GPT_Bin;

class app extends \Console
{   // https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt
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
        $v = $this->data($in, $cfg = cfg('gpt'));//print_r($v);
        if (!$v->txt && !$v->bin)
            return $this->a_u();
        $time = time();
        if ($load_bin = !$v->qtz && $v->bin) {
            [$v->settings, $state_dict] = GPT_Bin::load(self::$d[1] . "/bin/$v->bin.bin");
            $v->txt = $v->settings['dataset'];
        }
        $v->rnd && mt_srand($v->rnd);
        $gpt = new GPT_Run($v->settings);
        if ($load_bin)
            $gpt->params = $state_dict;
        $docs = $gpt->build_vocab(self::$d[1] . "/txt/$v->txt.txt");
        
        echo "num docs: " . count($docs) . "\n";
        echo "vocab size: $gpt->vocab_size\n";
        echo "num params: $gpt->n_params\n";
        echo "settings:\n" . $gpt->info(array_keys($cfg->default)) . "\n";

        if (!$load_bin) {
            foreach ($gpt->train($docs, $v->n_train, $v->learning_rate) as $i => $loss)
                echo "\rstep $i | loss $loss | seconds " . (time() - $time) . '     ';
            if ($v->qtz) {
                $qtz = $v->qtz == 4 ? GPT_Bin::Q_INT4 : ($v->qtz == 8 ? GPT_Bin::Q_INT8 : GPT_Bin::Q_FP32);
                $v->settings['dataset'] = $v->txt;
                GPT_Bin::save(self::$d[1] . "/bin/$v->bin.bin", $gpt->params, $v->settings, $qtz);
            }
        }
        foreach ($gpt->inference($v->temperature, $v->n_inference) as $i => $sample)
            echo "\nsample $i: $sample";
    }

    /** See Logos usage */
    function a_u() {
        $cfg = cfg('gpt');
        $gpt = new GPT_Run($cfg->default);
        echo strtr($cfg->usage, [
            '%list_1%' => var_export($cfg->short, true),
            '%list_2%' => $gpt->info(array_keys($cfg->default)),
        ]);
    }

    /** Generate the dataset (2do) */
    function a_g() {
        var_export(1);
    }
}
