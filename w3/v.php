<?php
/**
 * The most atomic way to train and inference a GPT in pure, dependency-free PHP.
 * This file is the complete algorithm.
 * Everything else is just efficiency.
 * 
 * Ported from Python by @karpathy
 */

ini_set('memory_limit', '988M');
require 'value.php';
require 'gpt_pack.php';

// ============================================================================
// Main execution
// ============================================================================

// Let there be order among chaos
mt_srand(37);

// Let there be an input dataset
$lines = array_map('trim', file('txt/names.txt'));
//$lines = array_map('trim', file('txt/math.txt')); // https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt
$docs = array_values(array_filter($lines, fn($l) => !empty($l)));
shuffle($docs);
echo "num docs: " . count($docs) . "\n";

// Let there be a Tokenizer
$unique_chars = array_unique(str_split(implode('', $docs)));
sort($unique_chars);
$chars = array_merge(['<BOS>'], $unique_chars);
$vocab_size = count($chars);
$stoi = $itos = [];
foreach ($chars as $i => $ch) {
    $stoi[$ch] = $i;
    $itos[$i] = $ch;
}

$BOS = $stoi['<BOS>'];
echo "vocab size: {$vocab_size}\n";

//var_export($itos);

// Initialize the parameters
$n_embd = 16; # 16 32
$n_head = 4;
$n_layer = 1;
$block_size = 8;
$head_dim = intval($n_embd / $n_head);
$n_params = $n_embd * ($block_size + 2 * $vocab_size + 12 * $n_layer * $n_embd);
echo "num params: $n_params\n";

if (0):
$state_dict = [
    'wte' => matrix($vocab_size, $n_embd),
    'wpe' => matrix($block_size, $n_embd),
    'lm_head' => matrix($vocab_size, $n_embd)
];
for ($i = 0; $i < $n_layer; $i++) {
    $state_dict["layer{$i}.attn_wq"] = matrix($n_embd, $n_embd);
    $state_dict["layer{$i}.attn_wk"] = matrix($n_embd, $n_embd);
    $state_dict["layer{$i}.attn_wv"] = matrix($n_embd, $n_embd);
    $state_dict["layer{$i}.attn_wo"] = matrix($n_embd, $n_embd, 0);
    $state_dict["layer{$i}.mlp_fc1"] = matrix(4 * $n_embd, $n_embd);
    $state_dict["layer{$i}.mlp_fc2"] = matrix($n_embd, 4 * $n_embd, 0);
}
// Let there be Adam
$learning_rate = 1e-2; # 1e-2 0.0003
$beta1 = 0.9;
$beta2 = 0.95;
$eps_adam = 1e-8;
$v = $m = array_fill(0, $n_params, 0.0);
date_default_timezone_set('Europe/Moscow');
echo date('Y-m-d H:i:s') . "\n";
$n_steps = 500; # 500
for ($step = 0; $step < $n_steps; $step++) { // Training loop
    // Take single document
    $doc = $docs[$step % count($docs)];
    $tokens = [$BOS];
    for ($i = 0; $i < strlen($doc); $i++)
        $tokens[] = $stoi[$doc[$i]];
    $tokens[] = $BOS;
    $n = min($block_size, count($tokens) - 1);
    
    // Forward pass
    $keys = $values = array_fill(0, $n_layer, $losses = []);
    
    for ($pos_id = 0; $pos_id < $n; $pos_id++) {
        $token_id = $tokens[$pos_id];
        $target_id = $tokens[$pos_id + 1];
        $logits = gpt($token_id, $pos_id, $keys, $values, $state_dict, $n_layer, $n_head, $head_dim);
        $probs = softmax_func($logits);
        $loss_t = $probs[$target_id]->log_val()->__neg();
        $losses[] = $loss_t;
    }
    
    $loss = new Value(0);
    foreach ($losses as $l)
        $loss = $loss->__add($l);
    $loss = $loss->__div($n);
    
    // Backward pass
    $loss->backward();
    
    // Adam optimizer update
    $lr_t = $learning_rate * (1 - $step / $n_steps);
    $i = 0;
    array_walk_recursive($state_dict, function($p) use (&$i, &$m, &$v, $lr_t, $beta1, $beta2, $step, $eps_adam) {
        $m[$i] = $beta1 * $m[$i] + (1 - $beta1) * $p->grad;
        $v[$i] = $beta2 * $v[$i] + (1 - $beta2) * ($p->grad ** 2);
        $m_hat = $m[$i] / (1 - pow($beta1, $step + 1));
        $v_hat = $v[$i] / (1 - pow($beta2, $step + 1));
        $p->data -= $lr_t * $m_hat / (sqrt($v_hat) + $eps_adam);
        $p->grad = 0;
        $i++;
    });
    
    printf("\rstep %4d / %4d | loss %.4f", $step + 1, $n_steps, $loss->data);
}
echo "\n\n";
$weights = $state_dict;
array_walk_recursive($weights, fn(&$v) => $v = $v->data);
//file_put_contents('gpt-name-vFP32.bin', GPT_Pack::encode($weights, GPT_Pack::Q_FP32));

else:
    //require 'param.php';
    $state_dict = GPT_pack::decode(file_get_contents('gpt-name-vFP32.bin'));
    //file_put_contents('zzz', var_export($state_dict,1));
    array_walk_recursive($state_dict, fn(&$v) => $v = new Value($v));

endif;

// Inference
$temperature = 0.6; # 0.6
echo "\n--- inference ---\n";
for ($i = 0; $i < 11; ) {
    echo "sample " . ++$i . ": ";
    $keys = $values = array_fill(0, $n_layer, []);
    $token_id = $BOS;
    for ($out = '', $pos_id = 0; $pos_id < $block_size; $pos_id++) {
        $logits = gpt($token_id, $pos_id, $keys, $values, $state_dict, $n_layer, $n_head, $head_dim);
        $scaled_logits = array_map(fn($l) => $l->__div($temperature), $logits);
        $probs = softmax_func($scaled_logits);
        $weights = array_map(fn($p) => $p->data, $probs);
        $token_id = random_choices(range(0, $vocab_size - 1), $weights);
        
        if ($token_id == $BOS)
            break;
        $out .= $itos[$token_id];
    }
    echo "$out\n";
}
