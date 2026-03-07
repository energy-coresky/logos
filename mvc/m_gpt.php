<?php

class m_gpt extends Model_m
{
    function run() {
        $mem = Shmem::open('segment');
        $mem->write(['losses' => [], 'stop' => 0]);
        $cfg = $_POST + cfg('gpt')->default;
        $gpt = new GPT_Run($cfg);
        $docs = $gpt->build_vocab($cfg['dataset']);
        $div = ceil($cfg['n_train'] / 120);
        $time = time();
        $mctime = microtime(true);
        $inference = $losses = $chunk = [];
        foreach ($gpt->train($docs, $cfg['n_train']) as $i => $loss) {
            $chunk[] = $loss;
            if (0 == $i % $div) {
                $losses[$i] = array_sum($chunk) / count($chunk);
                $chunk = [];
                if (time() > $time) {
                    $time = time();
                    if ($this->mem($mem, $cfg, $i, $mctime, $losses))
                        break;
                }
            }
        }
        $this->mem($mem, $cfg, $i, $mctime, $losses);
        sleep(2);
        $mem->close();
        if ($cfg['qtz']) {
            GPT_Bin::save($cfg['bin'], $gpt->params, $cfg, $cfg['qtz']);
        }
        return $cfg;
    }

    function progress() {
        $mem = Shmem::open('segment', false);
        $stop = $_POST['run'] != '1';
        return $mem->replace(false, ['losses' => [], 'stop' => (int)$stop]);
    }

    function mem($mem, $cfg, $i, $mctime, &$losses) {
        $data = $mem->read();
        $mem->write([
            'stop' => $data['stop'],
            'n_doc' => $n_doc = $cfg['batch_size'] * $i,
            'speed' => $n_doc / (microtime(true) - $mctime),
            'losses' => $data['losses'] + $losses,
        ]);
        $losses = [];
        return $data['stop'];
    }

    function form() {
        $form = new Form(cfg('form')->main);
        return $form->draw_form();
    }
}
