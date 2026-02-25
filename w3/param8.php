<?php

require 'gpt_pack.php';
$state_dict = GPT_pack::decode(file_get_contents('gpt-model-vINT8.bin'));
