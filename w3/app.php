<?php

namespace logos;
use Plan, GPT, GPT_Data;

class app extends \Console
{
    function __construct($argv = [], $found = []) {
        Plan::set('logos', fn() => parent::__construct($argv, $found));
    }

    function data($in, $out, &$v = null) {
        $cfg = cfg('gpt');
        $v = (object)$cfg->v;
        $this->args($in, $out, function (&$key, $val) use ($v, $cfg) {
            if ($set = isset($cfg->short[$key]))
                $key = $cfg->short[$key];
            $v->$key = $val;
        });
        return $out;
    }

    /** Run train &| inference */
    function a_run(...$in) {
        $main = $this->data($in, cfg('gpt')->main, $v);
        if (!$v->dataset && !$v->bin)
            return $this->a_usage();
        $v->rnd && mt_srand($v->rnd);
        $time = time();
        $gpt = new GPT($main, $v->bin);
        $docs = $gpt->build_vocab();
        echo "MAIN ARCHITECTURE:\n";
        foreach ($main as $key => $val)
            echo "  $key: $val\n";
        echo "  num docs: " . count($docs) . "\n";
        echo "  vocab size: $gpt->vocab_size\n";
        echo "  num params: $gpt->n_params\n";

        if (in_array('train', $in)) {
            echo "TRAIN:\n"; # math7 Theoretical Loss=0.597
            $train = $this->data($in, $gpt->init_weights());
            foreach ($train as $key => $val)
                echo "  $key: $val\n";
            foreach ($gpt->train($docs, (object)$train, $v->qtz) as $i => $y)
                echo "\r  step $i | loss $y->loss | seconds " . (time() - $time) . '     ';
            echo "\n";
        }

        if ($gpt->params) {
            $infer = $this->data($in, cfg('gpt')->infer);
            echo "INFERENCE:\n  temperature: $infer[temperature]";
            foreach ($gpt->inference($infer) as $i => $sample)
                echo "\n  sample $i: $sample";
        }
    }

    /** Run bin-file inference as chat */
    function a_chat(...$in) {
        $main = $this->data($in, cfg('gpt')->main, $v);
        if (!$v->bin)
            return $this->a_usage();
        $v->rnd && mt_srand($v->rnd);
        $gpt = new GPT($main, $v->bin);
        $gpt->build_vocab();
        foreach ($main as $key => $val)
            echo "  $key: $val\n";
        echo "  num params: $gpt->n_params\n";

        echo "INFERENCE:\n";
        $infer = $this->data($in, cfg('gpt')->infer);
        for (;1;) {
            $infer['prompt'] = trim(fgets(STDIN));
            foreach ($gpt->inference($infer) as $response)
                echo "  response: $infer[prompt]$response\n";
        }
    }

    /** Generate the datasets */
    function a_gen(...$in) { // names datasets - https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt
        $v = $this->data($in, $cfg = cfg('data'));
        if (method_exists($gen = new GPT_Data, $type = $v->type)) {
            $gen->$type($v->settings);
        } else echo strtr($cfg->usage, [
            '%list_1%' => var_export($cfg->short, true),
            '%list_2%' => var_export($v->settings, true),
        ]);
    }

    /** See Logos usage (how run the model) */
    function a_usage() {
        $cfg = cfg('gpt');
        echo strtr($cfg->usage, [
            '%list_1%' => var_export($cfg->short, true),
            '%list_2%' => var_export($cfg->main, true),
        ]);
    }
}
