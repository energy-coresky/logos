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
        print(11);
    }

    /** Test */
    function a_test() {
        print(11);
    }
}
