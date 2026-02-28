<?php

class GPT_Run extends GPT_Engine
{
    function __construct(array $prop = [])
    {
        ini_set('memory_limit', '1G');
        foreach ($prop as $k => $v)
            $this->$k = $v;
        $this->head_dim = intval($this->n_embd / $this->n_head);
    }

    function build_vocab(string $filename): array
    {
        $lines = array_map('trim', file($filename));
        $docs = array_values(array_filter($lines, fn($l) => !empty($l)));
        shuffle($docs);
        $unique_chars = array_unique(str_split(implode('', $docs)));
        sort($unique_chars);
        $chars = array_merge(['<BOS>'], $unique_chars);
        $this->vocab_size = count($chars);
        $this->n_params = $this->n_embd * (
            $this->block_size + 
            2 * $this->vocab_size + 
            12 * $this->n_layer * $this->n_embd
        );
        $this->itos = $this->stoi = [];
        foreach ($chars as $i => $ch) {
            $this->stoi[$ch] = $i;
            $this->itos[$i] = $ch;
        }
        $this->BOS = $this->stoi['<BOS>'];
        return $docs;
    }

    function init_weights(): void {
        $matrix = function ($nout, $nin, $std = 0.02) {
            // Box-Muller transform
            $gauss = function ($mean = 0, $std = 1) {
                static $spare = null;
                static $has_spare = false;
                if ($has_spare) {
                    $has_spare = false;
                    return $spare * $std + $mean;
                }
                $u = mt_rand() / mt_getrandmax(); $v = mt_rand() / mt_getrandmax();
                $u = $u < 1e-10 ? 1e-10 : $u;
                $mag = $std * sqrt(-2.0 * log($u));
                $spare = $mag * sin(2.0 * pi() * $v); $has_spare = true;
                return $mag * cos(2.0 * pi() * $v) + $mean;
            };
            for ($i = 0; $i < $nout; $i++)
                for ($j = 0; $j < $nin; $j++)
                    $ary[$i][$j] = $gauss(0, $std);
            return $ary;
        };

        $this->params = [
            'wte' => $matrix($this->vocab_size, $this->n_embd),
            'wpe' => $matrix($this->block_size, $this->n_embd),
            'lm_head' => $matrix($this->vocab_size, $this->n_embd)
        ];

        for ($i = 0; $i < $this->n_layer; $i++) {
            $this->params["layer{$i}.attn_wq"] = $matrix($this->n_embd, $this->n_embd);
            $this->params["layer{$i}.attn_wk"] = $matrix($this->n_embd, $this->n_embd);
            $this->params["layer{$i}.attn_wv"] = $matrix($this->n_embd, $this->n_embd);
            $this->params["layer{$i}.attn_wo"] = $matrix($this->n_embd, $this->n_embd, 0);
            $this->params["layer{$i}.mlp_fc1"] = $matrix(4 * $this->n_embd, $this->n_embd);
            $this->params["layer{$i}.mlp_fc2"] = $matrix($this->n_embd, 4 * $this->n_embd, 0);
        }

        $this->grads = $this->params;
        array_walk_recursive($this->grads, fn(&$v) => $v = 0.0);
        $this->m = $this->v = array_fill(0, $this->n_params, 0.0);
    }

    function _train(array $docs, int $n_steps, float $learning_rate = 1e-4, int $accum_steps = 1): Generator
    {
        $this->init_weights();
        $n_docs = count($docs);
        
        for ($step = $i_docs = 0; $step < $n_steps; ) {
            array_walk_recursive($this->grads, fn(&$v) => $v = 0.0);
            $batch_loss = 0.0;
            // --- Accumulation Loop ---
            for ($acc = 0; $acc < $accum_steps; $acc++) {
                $tokens = str_split($docs[$i_docs++ % $n_docs]);
                $tokens = array_merge([$this->BOS], array_map(fn($t) => $this->stoi[$t], $tokens), [$this->BOS]);
                if (!$n = min($this->block_size, count($tokens) - 1))
                    continue;

                // --- Forward pass ---
                $this->cache = $probs_all = [];
                $keys = $values = array_fill(0, $this->n_layer, []);
                
                for ($pos_id = 0; $pos_id < $n; $pos_id++) {
                    $token_id = $tokens[$pos_id];
                    $logits = $this->gpt($token_id, $pos_id, $keys, $values, false);
                    $probs_all[] = $this->softmax($logits);
                }

                // --- Backward pass ---
                // Важно: мы НЕ обнуляем градиенты тут! Они накапливаются.
                
                for ($pos_id = $n - 1; $pos_id >= 0; $pos_id--) {
                    $target_id = $tokens[$pos_id + 1];
                    $probs = $probs_all[$pos_id];
                    
                    $batch_loss += -log($probs[$target_id] + 1e-9);

                    $d_logits = $probs;
                    $d_logits[$target_id] -= 1.0;
                    
                    // Делим на $n (длина примера) И на $accum_steps (размер батча)
                    // Это усреднение градиента по всему "виртуальному батчу"
                    foreach ($d_logits as &$v)
                        $v /= ($n * $accum_steps);

                    $x_final = $this->cache['x_final'][$pos_id];
                    $this->accumulate_grad('lm_head', $d_logits, $x_final);
                    $d_x = $this->backward_linear_input($d_logits, $this->params['lm_head']);
                    
                    for ($li = $this->n_layer - 1; $li >= 0; $li--) {
                        $d_x = $this->backward_layer($d_x, $li, $pos_id, $keys, $values);
                    }

                    $token_id = $tokens[$pos_id];
                    for ($i = 0; $i < $this->n_embd; $i++) {
                        // Градиенты просто плюсуются
                        $this->grads['wte'][$token_id][$i] += $d_x[$i];
                        $this->grads['wpe'][$pos_id][$i] += $d_x[$i];
                    }
                }
            }

            // --- Adam Optimizer Update (делается 1 раз на $accum_steps примеров) ---
            // ... (код Adam без изменений) ...
            // Считаем норму
            $grad_norm = 0.0;
            array_walk_recursive($this->grads, function($p) use (&$grad_norm) {
                $grad_norm += $p ** 2;
            });
            $grad_norm = sqrt($grad_norm);
            $scale = $grad_norm > $this->grad_clip ? $this->grad_clip / $grad_norm : 1.0;
            // Linear decay
            $lr_t = $learning_rate * (1 - $step / $n_steps); 
            $idx = 0;
            foreach ($this->params as $name => &$matrix) {
                $rows = count($matrix);
                $cols = count($matrix[0]);
                for ($r = 0; $r < $rows; $r++) {
                    for ($c = 0; $c < $cols; $c++) {
                        $grad = $this->grads[$name][$r][$c] * $scale;
                        // Adam math
                        $this->m[$idx] = $this->beta1 * $this->m[$idx] + (1 - $this->beta1) * $grad;
                        $this->v[$idx] = $this->beta2 * $this->v[$idx] + (1 - $this->beta2) * ($grad ** 2);

                        $m_hat = $this->m[$idx] / (1 - pow($this->beta1, $step + 1));
                        $v_hat = $this->v[$idx] / (1 - pow($this->beta2, $step + 1));
                        // Update weight
                        $matrix[$r][$c] -= $lr_t * $m_hat / (sqrt($v_hat) + $this->eps_adam);
                        $idx++;
                    }
                }
            }
            yield ++$step => ($batch_loss / $accum_steps);
        }
    }



    function train(array $docs, int $n_steps, float $learning_rate = 1e-2): Generator
    {
        $this->init_weights();
        $n_docs = count($docs);
        
        for ($step = 0; $step < $n_steps; ) {
            $tokens = str_split($docs[$step % $n_docs]);
            $tokens = array_merge([$this->BOS], array_map(fn($t) => $this->stoi[$t], $tokens), [$this->BOS]);
            $n = min($this->block_size, count($tokens) - 1);

            // --- 1. Forward pass ---
            $this->cache = $probs_all = [];
            $keys = $values = array_fill(0, $this->n_layer, []);
            
            for ($pos_id = 0; $pos_id < $n; $pos_id++) {
                $token_id = $tokens[$pos_id];
                $logits = $this->gpt($token_id, $pos_id, $keys, $values, false);
                $probs_all[] = $this->softmax($logits);
            }

            // --- 2. Backward pass ---
            // Обнуляем градиенты перед накоплением
            array_walk_recursive($this->grads, fn(&$v) => $v = 0.0);
            
            $loss = 0.0;
            
            // Идем в обратном порядке по токенам
            for ($pos_id = $n - 1; $pos_id >= 0; $pos_id--) {
                $target_id = $tokens[$pos_id + 1];
                $probs = $probs_all[$pos_id];
                
                // Вычисляем Loss (для вывода)
                $loss += -log($probs[$target_id] + 1e-9);

                // Градиент функции потерь (Softmax + CrossEntropy)
                // d_loss/d_logits = probs - one_hot(target)
                $d_logits = $probs;
                $d_logits[$target_id] -= 1.0;
                
                // Нормализуем градиент на количество токенов (усреднение лосса)
                foreach ($d_logits as &$v) $v /= $n;

                // Backprop через LM Head (последний линейный слой)
                $x_final = $this->cache['x_final'][$pos_id];
                
                // Градиент по весам lm_head: dW = dy * x^T
                $this->accumulate_grad('lm_head', $d_logits, $x_final);
                
                // Градиент на вход x (d_x), чтобы передать в трансформер: d_x = dy * W
                $d_x = $this->backward_linear_input($d_logits, $this->params['lm_head']);
                
                // Backprop через слои Трансформера (от последнего к первому)
                for ($li = $this->n_layer - 1; $li >= 0; $li--) {
                    $d_x = $this->backward_layer($d_x, $li, $pos_id, $keys, $values);
                }
                
                // Backprop через начальную нормализацию (Init Norm)
                // В упрощенной реализации rmsnorm мы считаем scale константой для backward
                // или применяем точную формулу. Здесь берем упрощение: d_x_in = d_x_out * scale.
                // Для полноценной реализации нужен backward_rmsnorm, но для обучения "и так сойдет"
                // часто достаточно простой передачи градиента, если scale близок к 1.

                // Backprop через Embeddings
                // x = wte[token] + wpe[pos]
                // Градиенты просто прибавляются к соответствующим строкам матриц wte и wpe
                $token_id = $tokens[$pos_id];
                for ($i = 0; $i < $this->n_embd; $i++) {
                    $this->grads['wte'][$token_id][$i] += $d_x[$i];
                    $this->grads['wpe'][$pos_id][$i] += $d_x[$i];
                }
            }
            $loss /= $n;

            // --- 3. Adam Optimizer Update ---
            // Считаем норму градиента (длину вектора всех градиентов)
            $grad_norm = 0.0;
            array_walk_recursive($this->grads, function($p) use (&$grad_norm) {
                $grad_norm += $p ** 2;
            });
            $grad_norm = sqrt($grad_norm);
            // Если норма слишком большая - масштабируем (клиппинг)
            $scale = $grad_norm > $this->grad_clip ? $this->grad_clip / $grad_norm : 1.0;
            $lr_t = $learning_rate * (1 - $step / $n_steps); // Linear decay
            $idx = 0;
            foreach ($this->params as $name => &$matrix) {
                $rows = count($matrix);
                $cols = count($matrix[0]);
                for ($r = 0; $r < $rows; $r++) {
                    for ($c = 0; $c < $cols; $c++) {
                        $grad = $this->grads[$name][$r][$c] * $scale;
                        // Adam math
                        $this->m[$idx] = $this->beta1 * $this->m[$idx] + (1 - $this->beta1) * $grad;
                        $this->v[$idx] = $this->beta2 * $this->v[$idx] + (1 - $this->beta2) * ($grad ** 2);

                        $m_hat = $this->m[$idx] / (1 - pow($this->beta1, $step + 1));
                        $v_hat = $this->v[$idx] / (1 - pow($this->beta2, $step + 1));

                        // Update weight
                        $matrix[$r][$c] -= $lr_t * $m_hat / (sqrt($v_hat) + $this->eps_adam);
                        $idx++;
                    }
                }
            }
            yield ++$step => $loss;
        }
    }

    function inference(float $temperature = 0.6, int $count = 10): Generator
    {
        for ($i = 0; $i < $count; ) {
            $keys = $values = array_fill(0, $this->n_layer, []);
            $token_id = $this->BOS;
            $out = '';
            for ($pos_id = 0; $pos_id < $this->block_size; $pos_id++) {
                $logits = $this->gpt($token_id, $pos_id, $keys, $values);
                
                // Apply temperature
                $scaled_logits = array_map(fn($l) => $l / $temperature, $logits);
                $weights = $this->softmax($scaled_logits);
                
                $token_id = $this->random_choices(range(0, $this->vocab_size - 1), $weights);
                if ($token_id == $this->BOS)
                    break;
                $out .= $this->itos[$token_id];
            }
            yield ++$i => $out;
        }
    }
}
