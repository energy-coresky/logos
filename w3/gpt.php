<?php

class GPT extends GPT_Engine
{
    private $binary;

    function __construct(&$main = null, $binary = '')
    {
        ini_set('memory_limit', '1G');
        if ('' !== $binary) {
            $main = GPT_Bin::load($this->binary = $binary, $this->params);
        } else {
            $main += cfg('gpt')->main;
        }
        foreach ($main as $k => $v)
            $this->$k = $v;
        $this->head_dim = intval($this->n_embd / $this->n_head);
        $this->scale_head = 1.0 / sqrt($this->head_dim);
    }

    function &build_vocab(): array
    {
        $lines = array_map('trim', explode("\n", Plan::txt_g("$this->dataset.txt")));
        $docs = array_values(array_filter($lines, fn($l) => !empty($l)));
        shuffle($docs);
        $unique_chars = array_unique(str_split(implode('', $docs)));
        sort($unique_chars);
        $chars = array_merge(['<BOS>'], $unique_chars);
        $this->vocab_size = count($chars);
        $this->itos = $this->stoi = [];
        foreach ($chars as $i => $ch) {
            $this->stoi[$ch] = $i;
            $this->itos[$i] = $ch;
        }
        $this->BOS = $this->stoi['<BOS>'];
        $this->n_params = $this->n_embd * (
            1 + $this->block_size +
            2 * ($this->n_layer + $this->vocab_size) +
            12 * $this->n_layer * $this->n_embd
        );
        return $docs;
    }

    // Box-Muller transform
    function gauss($mean = 0, $std = 1): void {
        static $spare = null;
        static $has_spare = false;
        if ($has_spare) {
            $has_spare = false;
            return $spare * $std + $mean;
        }
        $u = mt_rand() / mt_getrandmax();
        $v = mt_rand() / mt_getrandmax();
        $u = $u < 1e-10 ? 1e-10 : $u;
        $mag = $std * sqrt(-2.0 * log($u));
        $spare = $mag * sin(2.0 * pi() * $v);
        $has_spare = true;
        return $mag * cos(2.0 * pi() * $v) + $mean;
    }

    function init_weights(): array {
        if ($this->params)
            return GPT_Bin::load("$this->binary.adam", $this->m, $this->v);

        $matrix = function ($rows, $cols, $std = 0.02) {
            for ($i = 0; $i < $rows; $i++)
                for ($j = 0; $j < $cols; $j++)
                    $ary[$i][$j] = $this->gauss(0, $std);
            return $ary;
        };

        $this->params = [
            'wte' => $matrix($this->vocab_size, $this->n_embd),
            'wpe' => $matrix($this->block_size, $this->n_embd),
            'lm_head' => $matrix($this->vocab_size, $this->n_embd),
            'init_norm' => array_fill(0, $this->n_embd, 1.0), # params for rmsnorn
        ];
        for ($i = 0; $i < $this->n_layer; $i++) {
            $this->params["layer{$i}.attn_wq"] = $matrix($this->n_embd, $this->n_embd);
            $this->params["layer{$i}.attn_wk"] = $matrix($this->n_embd, $this->n_embd);
            $this->params["layer{$i}.attn_wv"] = $matrix($this->n_embd, $this->n_embd);
           #$this->params["layer{$i}.attn_wo"] = $matrix($this->n_embd, $this->n_embd, 0);
            $this->params["layer{$i}.attn_wo"] = $matrix($this->n_embd, $this->n_embd, 0.02 / sqrt(2 * $this->n_layer));
            $this->params["layer{$i}.mlp_fc1"] = $matrix(4 * $this->n_embd, $this->n_embd);
            $this->params["layer{$i}.mlp_fc2"] = $matrix($this->n_embd, 4 * $this->n_embd, 0);
            $this->params["pre_att_{$i}"] = array_fill(0, $this->n_embd, 1.0); # params for rmsnorn
            $this->params["pre_mlp_{$i}"] = array_fill(0, $this->n_embd, 1.0); # params for rmsnorn
        }
        $this->m = $this->v = array_fill(0, $this->n_params, 0.0);
        return cfg('gpt')->train;
    }

    function train(array &$docs, $train = [], $qtz = 0): Generator {
        $y = (object)($train + $this->init_weights());
        $this->grads = $this->params;
        $n_docs = count($docs);
        $i_docs = 0;
        $min_lr = y->learning_rate * 0.1; // $min_lr обычно ставят 10% от y->learning_rate или 1e-5
        for ($step =& $y->epoch_from; $step < $y->n_epoch; ) {
            array_walk_recursive($this->grads, fn(&$v) => $v = 0.0);
            $batch_loss = 0.0;
            for ($b = 0; $b < $y->batch_size; $b++) {
                $tokens = str_split($docs[$i_docs++ % $n_docs]);
                $tokens = array_merge([$this->BOS], array_map(fn($t) => $this->stoi[$t], $tokens), [$this->BOS]);
                $batch_loss += $this->train_one_doc($tokens, $y->batch_size);
            }
            // --- Adam Optimizer Update ---
            $grad_norm = 0.0;
            array_walk_recursive($this->grads, function($p) use (&$grad_norm) {
                $grad_norm += $p ** 2;
            });
            $grad_norm = sqrt($grad_norm);
            $scale_clip = $grad_norm > $y->grad_clip ? $y->grad_clip / $grad_norm : 1.0;
            // Linear decay
          # $lr_t = y->learning_rate * (1 - $step / $y->n_epoch);
            # Cosine decay
            if ($step < $y->warmup_epoch) {
                $lr_t = y->learning_rate * ($step / $y->warmup_epoch);
            } else {
                $decay_ratio = ($step - $y->warmup_epoch) / ($y->n_epoch - $y->warmup_epoch);
                $coeff = 0.5 * (1 + cos(pi() * $decay_ratio));
                $lr_t = $min_lr + (y->learning_rate - $min_lr) * $coeff;
            }
            // --- Adam Optimizer Update ---
            $bias_correction1 = 1 - pow($y->beta1, $step + 1);
            $bias_correction2 = 1 - pow($y->beta2, $step + 1);
            $idx = 0;
            foreach ($this->params as $name => &$p) {
                $grads_block =& $this->grads[$name];
                $is_matrix = isset($p[0]) && is_array($p[0]);
                $rows = count($p);
                for ($r = 0; $r < $rows; $r++) {
                    if ($is_matrix) {
                        $row_grad =& $grads_block[$r]; 
                        $cols = count($p[0]);
                        for ($c = 0; $c < $cols; $c++) {
                            $grad = $row_grad[$c] * $scale_clip;
                            $this->m[$idx] = $y->beta1 * $this->m[$idx] + (1 - $y->beta1) * $grad;
                            $this->v[$idx] = $y->beta2 * $this->v[$idx] + (1 - $y->beta2) * ($grad ** 2);
                            $m_hat = $this->m[$idx] / $bias_correction1;
                            $v_hat = $this->v[$idx] / $bias_correction2;
                            $p[$r][$c] -= $lr_t * $m_hat / (sqrt($v_hat) + $y->eps_adam);
                            $idx++;
                        }
                    } else {
                        $grad = $grads_block[$r] * $scale_clip;
                        $this->m[$idx] = $y->beta1 * $this->m[$idx] + (1 - $y->beta1) * $grad;
                        $this->v[$idx] = $y->beta2 * $this->v[$idx] + (1 - $y->beta2) * ($grad ** 2);
                        $m_hat = $this->m[$idx] / $bias_correction1;
                        $v_hat = $this->v[$idx] / $bias_correction2;
                        $p[$r] -= $lr_t * $m_hat / (sqrt($v_hat) + $y->eps_adam);
                        $idx++;
                    }
                }
            }
            $y->loss = $batch_loss / $y->batch_size;
            yield ++$step => $y;
            if ($y->checkpoint)
                $this->save_bin($y, $qtz);
        }
        $this->save_bin($y, $qtz);
    }

    private function save_bin($y, $qtz) {
        $y->checkpoint = false;
        if ($y->bin_out && $qtz) {
            GPT_Bin::save($y->bin_out, $this->params, $qtz, [
                'n_embd'     => $this->n_embd,
                'n_head'     => $this->n_head,
                'n_layer'    => $this->n_layer,
                'block_size' => $this->block_size,
                'dataset'    => $this->dataset,
            ]);
            GPT_Bin::save("$y->bin_out.adam", [$this->m, $this->v], $qtz, (array)$y);
        }
    }

    private function train_one_doc(array $tokens, $batch_size): float {
        $n = min($this->block_size, count($tokens) - 1);
        $this->cache = $probs_all = [];
        $keys = $values = array_fill(0, $this->n_layer, []);
        $loss = 0.0;

        for ($pos_id = 0; $pos_id < $n; $pos_id++) {
            $token_id = $tokens[$pos_id];
            $logits = $this->gpt($token_id, $pos_id, $keys, $values, false);
            $probs_all[] = $this->softmax($logits);
        }

        for ($pos_id = $n - 1; $pos_id >= 0; $pos_id--) {
            $target_id = $tokens[$pos_id + 1];
            $d_logits = $probs_all[$pos_id];
            $loss += -log($d_logits[$target_id] + 1e-9);
            $d_logits[$target_id] -= 1.0;
            foreach ($d_logits as &$v)
                $v /= ($n * $batch_size);

            $x_final = $this->cache['x_final'][$pos_id];
            $this->accumulate_grad('lm_head', $d_logits, $x_final);
            $d_x = $this->backward_linear_input($d_logits, $this->params['lm_head']);

            for ($li = $this->n_layer - 1; $li >= 0; $li--) {
                $d_x = $this->backward_layer($d_x, $li, $pos_id, $keys, $values);
            }

            $d_x = $this->backward_rmsnorm($d_x, 'init_norm', $pos_id);

            $token_id = $tokens[$pos_id];
            for ($i = 0; $i < $this->n_embd; $i++) {
                $this->grads['wte'][$token_id][$i] += $d_x[$i];
                $this->grads['wpe'][$pos_id][$i] += $d_x[$i];
            }
        }
        return $loss / $n;
    }

    function inference(float $temperature = 0.6, int $count = 10, string $prompt = ''): Generator {
        $tokens = [$this->BOS];
        foreach (str_split($prompt) as $char) {
            if (isset($this->stoi[$char]))
                $tokens[] = $this->stoi[$char];
        }

        for ($i = 0; $i < $count; ) {
            $keys = $values = array_fill(0, $this->n_layer, []);
            $pos_id = 0;
            foreach ($tokens as $token_id)
                $logits = $this->gpt($token_id, $pos_id++, $keys, $values);

            for ($out = ''; $pos_id < $this->block_size; ) {
                $weights = $this->softmax(array_map(fn($l) => $l / $temperature, $logits));
                $token_id = $this->random_choices(range(0, $this->vocab_size - 1), $weights);
                if ($token_id == $this->BOS)
                    break;
                $out .= $this->itos[$token_id];
                $logits = $this->gpt($token_id, $pos_id++, $keys, $values);
            }
            yield ++$i => $out;
        }
    }

    function random_choices($population, $weights) {
        $rand = mt_rand() / mt_getrandmax() * array_sum($weights);
        $cumulative = 0;
        foreach ($population as $i => $item) {
            $cumulative += $weights[$i];
            if ($rand <= $cumulative)
                return $item;
        }
        return $population[count($population) - 1];
    }
}
