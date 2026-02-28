<?php

class GPT_Engine
{
    // Architecture Params
    public int $n_embd = 32;
    public int $n_head = 4;
    public int $n_layer = 2;
    public int $block_size = 8;
    public int $head_dim;

    // Model State
    public array $params = [];
    public array $grads = []; // Градиенты (симметричны params)
    public int $n_params = 0;
    public int $vocab_size = 0;

    // Optimizer State (Adam)
    protected array $m = [];
    protected array $v = [];

    public float $beta1 = 0.9;
    public float $beta2 = 0.95;
    public float $eps_adam = 1e-8;
    public float $grad_clip = 1.0; // Максимальное значение градиента

    // Tokenizer
    protected array $stoi = [];
    protected array $itos = [];
    protected int $BOS = 0;

    // Runtime Cache for Backward pass
    // Храним данные Forward прохода, индексированные по позиции и слою
    protected array $cache = [];

    // Define the model architecture
    public function gpt($token_id, $pos_id, &$keys, &$values, $is_inference = true) {
        $tok_emb = $this->params['wte'][$token_id];
        $pos_emb = $this->params['wpe'][$pos_id];
        
        $x = [];
        for ($i = 0; $i < $this->n_embd; $i++)
            $x[] = $tok_emb[$i] + $pos_emb[$i];
        $x = $this->rmsnorm($x, 'init_norm');

        // Кэшируем вход для эмбеддингов (нужно для градиентов wte/wpe)
        $is_inference or $this->cache['x_emb'][$pos_id] = $x;

        // Нормализация перед слоями (обычно делается, но для упрощения опустим глобальную норму, оставим внутри блоков)
        for ($li = 0; $li < $this->n_layer; $li++) {
            // --- Attention Block ---
            $x_in_att = $x; // Сохраняем для residual связи
            
            // Нормализация. Для backward нам нужно сохранить результат нормализации
            // Label используется только для отладки или если бы мы кэшировали внутри rmsnorm
            $x_norm = $this->rmsnorm($x, "pre_att_{$li}_{$pos_id}");

            // Проекции Q, K, V
            $q = $this->linear_layer($x_norm, $this->params["layer{$li}.attn_wq"]);
            $k = $this->linear_layer($x_norm, $this->params["layer{$li}.attn_wk"]);
            $v = $this->linear_layer($x_norm, $this->params["layer{$li}.attn_wv"]);
            
            // Сохраняем K и V в глобальный кэш для Attention
            $keys[$li][] = $k;
            $values[$li][] = $v;

            // --- Multi-Head Attention Logic ---
            $attn_weights_cache = []; // веса внимания для backward
            $x_attn = array_fill(0, $this->n_embd, 0.0);
            
            for ($h = 0; $h < $this->n_head; $h++) {
                $hs = $h * $this->head_dim;
                $q_h = array_slice($q, $hs, $this->head_dim);
                
                $attn_logits = [];
                // Считаем attention scores со всеми предыдущими токенами
                foreach ($keys[$li] as $ki) {
                    $k_h = array_slice($ki, $hs, $this->head_dim);
                    $sum = 0.0;
                    for ($j = 0; $j < $this->head_dim; $j++)
                        $sum += $q_h[$j] * $k_h[$j];
                    $attn_logits[] = $sum / sqrt($this->head_dim);
                }
                
                $attn_weights = $this->softmax($attn_logits);
                // Кэш для backward
                $is_inference or $attn_weights_cache[$h] = $attn_weights;

                // Взвешенная сумма Values
                $head_out = array_fill(0, $this->head_dim, 0.0);
                foreach ($values[$li] as $t => $vi) {
                    $v_h = array_slice($vi, $hs, $this->head_dim);
                    $w = $attn_weights[$t];
                    for ($j = 0; $j < $this->head_dim; $j++)
                        $head_out[$j] += $w * $v_h[$j];
                }

                // Собираем головы вместе
                for ($j = 0; $j < $this->head_dim; $j++)
                    $x_attn[$hs + $j] = $head_out[$j];
            }
            
            // Проекция выхода Attention
            $x_proj = $this->linear_layer($x_attn, $this->params["layer{$li}.attn_wo"]);
            // Residual connection
            for ($i = 0; $i < $this->n_embd; $i++)
                $x[$i] = $x_in_att[$i] + $x_proj[$i];

            // --- MLP Block ---
            $x_in_mlp = $x; // Residual
            $x_norm_mlp = $this->rmsnorm($x, "pre_mlp_{$li}_{$pos_id}");
            $h = $this->linear_layer($x_norm_mlp, $this->params["layer{$li}.mlp_fc1"]);
            
            // Activation (ReLU^2) + Cache
            for ($i = 0; $i < count($h); $i++) {
                $val = $h[$i];
                $relu = $val > 0 ? $val : 0;
                $is_inference or $this->cache["relu_{$li}_{$pos_id}"][$i] = $relu; // Кэш для backward ReLU
                $h[$i] = $relu * $relu;
            }
            $h_proj = $this->linear_layer($h, $this->params["layer{$li}.mlp_fc2"]);
            
            // Residual connection
            for ($i = 0; $i < $this->n_embd; $i++)
                $x[$i] = $x_in_mlp[$i] + $h_proj[$i];

            if ($is_inference)
                continue;

            $this->cache["x_norm_attn_{$li}_{$pos_id}"] = $x_norm;         # вход в Attention.
            // Cache Q, K, V для backward
            // K и V сохраняются в массивы $keys/$values, их потом можно достать оттуда по индексу $pos_id
            $this->cache["q_{$li}_{$pos_id}"] = $q;                        # вектор запроса Q.
            $this->cache["attn_w_{$li}_{$pos_id}"] = $attn_weights_cache;  # веса внимания для всего слоя (после цикла по головам)
            $this->cache["mlp_h_{$li}_{$pos_id}"] = $h;                    # вход во второй MLP слой.
            $this->cache["x_norm_mlp_{$li}_{$pos_id}"] = $x_norm_mlp;      # вход в MLP.
            $this->cache["attn_out_{$li}_{$pos_id}"] = $x_attn;            # выход Attention (перед проекцией wo).
        }

        $is_inference or $this->cache['x_final'][$pos_id] = $x;
        // Final Head
        return $this->linear_layer($x, $this->params['lm_head']);
    }

    /**
     * Полный Backward проход для одного слоя Трансформера.
     * 
     * @param array $d_x Градиент, приходящий с выхода следующего слоя (или LM Head).
     * @param int $li Индекс слоя.
     * @param int $pos_id Текущая позиция токена (для работы с KV Cache).
     * @param array $keys Ссылка на массив ключей (для вычисления градиентов K).
     * @param array $values Ссылка на массив значений (для вычисления градиентов V).
     * @return array Градиент для передачи на вход предыдущего слоя (или эмбеддингов).
     */
    protected function backward_layer(array $d_x, int $li, int $pos_id, array &$keys, array &$values): array
    {
        // --- 1. MLP Block Backward ---
        $d_x_mlp = $d_x; 
        $d_x_res = $d_x; 

        // Backward через fc2
        $h_in = $this->cache["mlp_h_{$li}_{$pos_id}"]; 
        $this->accumulate_grad("layer{$li}.mlp_fc2", $d_x_mlp, $h_in);
        $d_h = $this->backward_linear_input($d_x_mlp, $this->params["layer{$li}.mlp_fc2"]);

        // Backward через ReLU^2
        $relu_cache = $this->cache["relu_{$li}_{$pos_id}"];
        $d_fc1_out = [];
        for ($i = 0; $i < count($d_h); $i++) {
            $deriv = $relu_cache[$i] > 0 ? 2 * $relu_cache[$i] : 0; 
            $d_fc1_out[$i] = $d_h[$i] * $deriv;
        }

        // Backward через fc1
        $x_norm_mlp = $this->cache["x_norm_mlp_{$li}_{$pos_id}"];
        $this->accumulate_grad("layer{$li}.mlp_fc1", $d_fc1_out, $x_norm_mlp);
        $d_x_norm_mlp = $this->backward_linear_input($d_fc1_out, $this->params["layer{$li}.mlp_fc1"]); // Исправил индекс params на $li

        // Суммируем градиенты MLP и Residual
        $d_x_after_attn = [];
        for ($i = 0; $i < $this->n_embd; $i++)
            $d_x_after_attn[$i] = $d_x_norm_mlp[$i] + $d_x_res[$i];

        // --- 2. Attention Block Backward ---
        $d_attn_out = $d_x_after_attn; 
        
        // Backward через attn_wo (Output Projection)
        $attn_ctx = $this->cache["attn_out_{$li}_{$pos_id}"]; 
        $this->accumulate_grad("layer{$li}.attn_wo", $d_attn_out, $attn_ctx);
        $d_attn_ctx = $this->backward_linear_input($d_attn_out, $this->params["layer{$li}.attn_wo"]);

        // Градиенты для Q, K, V проекций
        $d_q_total = array_fill(0, $this->n_embd, 0.0);
        
        // Получаем кэш весов внимания для текущего шага
        $attn_weights_cache = $this->cache["attn_w_{$li}_{$pos_id}"];
        
        // Определяем, сколько токенов участвовало в внимании на этом шаге
        // Это важно, чтобы не вылезти за границы массивов
        $limit = count($attn_weights_cache[0] ?? []); 

        for ($h = 0; $h < $this->n_head; $h++) {
            $hs = $h * $this->head_dim;
            $d_head_out = array_slice($d_attn_ctx, $hs, $this->head_dim);
            
            $attn_w = $attn_weights_cache[$h]; // Веса [0..limit-1]
            
            // --- 1. Gradient on Values (d_v) and Logits (d_s) ---
            // Нам нужно накопить градиенты d_v для всех t, чтобы обновить WV
            // И посчитать градиенты для Logits (d_s)
            
            $d_attn_logits = array_fill(0, $limit, 0.0);
            
            // Итерируемся только по тем токенам, которые были в attention (от 0 до limit-1)
            for ($t = 0; $t < $limit; $t++) {
                $v_h = array_slice($values[$li][$t], $hs, $this->head_dim);
                $w = $attn_w[$t];
                
                // Градиент для Logits (d_s)
                for ($j = 0; $j < $this->head_dim; $j++) {
                    $d_attn_logits[$t] += $d_head_out[$j] * $v_h[$j];
                }
            }

            // --- 2. Softmax Backward ---
            $sum_p_ds = 0.0;
            for ($t = 0; $t < $limit; $t++) {
                $sum_p_ds += $attn_w[$t] * $d_attn_logits[$t];
            }
            
            $d_scores = [];
            for ($t = 0; $t < $limit; $t++) {
                $d_scores[$t] = $attn_w[$t] * ($d_attn_logits[$t] - $sum_p_ds);
            }

            // --- 3. Gradient on Queries and Keys (and final V update) ---
            $q_curr = array_slice($this->cache["q_{$li}_{$pos_id}"], $hs, $this->head_dim);
            $scale = 1.0 / sqrt($this->head_dim);
            
            for ($t = 0; $t < $limit; $t++) {
                $k_t = array_slice($keys[$li][$t], $hs, $this->head_dim);
                $score_t = $d_scores[$t];
                
                // d_q (накапливаем, так как Q использовался для всех t)
                for ($j = 0; $j < $this->head_dim; $j++) {
                    $d_q_total[$hs + $j] += $score_t * $scale * $k_t[$j];
                }
                
                // d_k и d_v для позиции t.
                // Нам нужно обновить веса WK и WV.
                // Для этого нам нужен вход x_norm_attn для позиции t.
                // Это и есть BPTT (Backpropagation Through Time).
                
                $x_t = $this->cache["x_norm_attn_{$li}_{$t}"]; // Вход в проекции K и V на шаге t
                
                // Градиент K для позиции t
                // d_k_h[j] = score * q[j]
                for ($j = 0; $j < $this->head_dim; $j++) {
                    $grad_k = $score_t * $scale * $q_curr[$j];
                    
                    // Обновляем WK (строка hs+j, все столбцы)
                    // dWK = d_k * x_t^T
                    $row_idx = $hs + $j;
                    foreach ($x_t as $col => $x_val) {
                        $this->grads["layer{$li}.attn_wk"][$row_idx][$col] += $grad_k * $x_val;
                    }
                }
                
                // Градиент V для позиции t
                // d_v_h[j] = attn_w[t] * d_out[j]
                for ($j = 0; $j < $this->head_dim; $j++) {
                    $grad_v = $attn_w[$t] * $d_head_out[$j];
                    
                    // Обновляем WV
                    $row_idx = $hs + $j;
                    foreach ($x_t as $col => $x_val) {
                        $this->grads["layer{$li}.attn_wv"][$row_idx][$col] += $grad_v * $x_val;
                    }
                }
            }
        }
        
        // --- 4. Backprop through Projections ---
        
        // Q projection (используем накопленный d_q_total)
        $x_norm_attn = $this->cache["x_norm_attn_{$li}_{$pos_id}"];
        $this->accumulate_grad("layer{$li}.attn_wq", $d_q_total, $x_norm_attn);
        $d_x_q = $this->backward_linear_input($d_q_total, $this->params["layer{$li}.attn_wq"]);

        // K and V projections градиенты мы уже обновили вручную в цикле выше (BPTT),
        // но нам нужно получить градиент на вход x_norm_attn для передачи дальше.
        // Для текущей позиции (pos_id) K и V были созданы здесь.
        
        // Восстанавливаем d_k и d_v для текущей позиции (t = pos_id = limit-1)
        // Это нужно, чтобы посчитать dx. 
        // Упрощение: возьмем градиент только для текущей позиции.
        $t_last = $pos_id;
        $d_k_curr = array_fill(0, $this->n_embd, 0.0);
        $d_v_curr = array_fill(0, $this->n_embd, 0.0);
        
        for ($h = 0; $h < $this->n_head; $h++) {
            $hs = $h * $this->head_dim;
            $score_last = $d_scores[$limit-1]; // score for last token
            $q_curr = array_slice($this->cache["q_{$li}_{$pos_id}"], $hs, $this->head_dim);
            $w_last = $attn_weights_cache[$h][$limit-1];
            $d_head_out = array_slice($d_attn_ctx, $hs, $this->head_dim);
            
            for ($j = 0; $j < $this->head_dim; $j++) {
                $d_k_curr[$hs+$j] = $score_last * $scale * $q_curr[$j];
                $d_v_curr[$hs+$j] = $w_last * $d_head_out[$j];
            }
        }

        $d_x_k = $this->backward_linear_input($d_k_curr, $this->params["layer{$li}.attn_wk"]);
        $d_x_v = $this->backward_linear_input($d_v_curr, $this->params["layer{$li}.attn_wv"]);

        // Суммируем градиенты от Q, K, V
        $d_x_norm_attn = [];
        for ($i = 0; $i < $this->n_embd; $i++)
            $d_x_norm_attn[$i] = $d_x_q[$i] + $d_x_k[$i] + $d_x_v[$i];
            
        // Residual connection
        $d_input = $d_x_after_attn; 
        for ($i = 0; $i < $this->n_embd; $i++)
            $d_input[$i] += $d_x_norm_attn[$i];

        return $d_input;
    }
    
    // Хелпер: накопление градиента для матрицы W (dW = dy * dx^T)
    protected function accumulate_grad(string $key, array $dy, array $x) {
        $w_rows = count($dy);
        $w_cols = count($x);
        // Проходим по строкам W
        for ($i = 0; $i < $w_rows; $i++) {
            for ($j = 0; $j < $w_cols; $j++)
                $this->grads[$key][$i][$j] += $dy[$i] * $x[$j];
        }
    }
    
    // Хелпер: расчет градиента на вход Linear слоя (dx = dy * W^T)
    protected function backward_linear_input(array $dy, array $w) {
        $dx = array_fill(0, count($w[0]), 0.0);
        for ($i = 0; $i < count($w); $i++) { // rows of W (out features)
            for ($j = 0; $j < count($w[0]); $j++) // cols of W (in features)
                $dx[$j] += $dy[$i] * $w[$i][$j];
        }
        return $dx;
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

    function linear_layer($x, $w) {
        $result = [];
        foreach ($w as $wo) {
            $sum = 0.0;
            foreach ($wo as $i => $wi)
                $sum += $wi * $x[$i];
            $result[] = $sum;
        }
        return $result;
    }

    function softmax($logits) {
        $max_val = max($logits);
        $exps = array_map(fn($val) => exp($val - $max_val), $logits);
        $total = array_sum($exps);
        return array_map(fn($e) => $e / $total, $exps);
    }

    function rmsnorm($x, $label = '') {
        $ms = 0.0;
        foreach ($x as $xi)
            $ms += $xi * $xi;
        $ms /= count($x);
        $scale = 1.0 / sqrt($ms + 1e-5);
        return array_map(fn($xi) => $xi * $scale, $x);
    }
}
