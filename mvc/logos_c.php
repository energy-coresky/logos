<?php

class logos_c extends Controller
{
    function head_y($action) {
        $vars = Plan::cfg_gq('logos_vars.txt');
        SKY::ghost('w', $vars, function ($s) {
            Plan::cfg_p(['logos', 'logos_vars.txt'], $s);
        });
    }

    function tail_y() {
        if (!MVC::$layout)
            return;
        '_logos?ware' == URI or SKY::w('last_link', URI);
        $y = parent::tail_y();
        Plan::head('', '~/w/logos/chart.umd.min.js'); # <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        return $y + ['chart' => json_encode(cfg('chart')),];
    }

    function a_run() {
        SKY::w('last_sand', $this->_2);
        return ['list' => implode('', array_map(function ($v) {
            return a(substr(basename($v, '.txt'), 6), ['sky.d.preset($(this))']);
        }, Plan::cfg_b("mvc/preset/$this->_2-*"))),
          'form'  => $this->m_gpt->form(),
        ];
    }

    function j_preset() {
        return [
            'fn' => $fn = 'mvc/preset/' . $_POST['n'] . '.txt',
            'ary' => explode('~', Plan::cfg_g($fn)),
        ];
    }

    function a_ware() {
        '_logos?ware' != $this->w_last_link or $this->w_last_link = '';
        jump($this->w_last_link ?: '_logos?gpt=0');
    }

    function j_run() {
        set_time_limit(0);
        $cfg = $this->m_gpt->run();
        json(['cfg' => $cfg]);
    }

    function j_progress() {
        MVC::$layout = '';
        header('Content-Type: application/json; charset=' . ENC);
        echo $this->m_gpt->progress();
    }

    function a_gen() {
    }

    function a_data() {
    }
}
