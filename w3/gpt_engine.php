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
    public float $learning_rate = 1e-4;
    public int $batch_size = 36;

    // Tokenizer
    protected array $stoi = [];
    protected array $itos = [];
    protected int $BOS = 0;

    // Runtime Cache for Backward pass
    // Храним данные Forward прохода, индексированные по позиции и слою
    protected array $cache = [];
    protected $scale_head;

    // Define the model architecture
    public function gpt($token_id, $pos_id, &$keys, &$values, $is_inference = true) {
        $tok_emb = $this->params['wte'][$token_id];
        $pos_emb = $this->params['wpe'][$pos_id];
        $x = [];
        for ($i = 0; $i < $this->n_embd; $i++)
            $x[] = $tok_emb[$i] + $pos_emb[$i];
        $x = $this->rmsnorm($x, 'init_norm', $pos_id);

        $is_inference or $this->cache['x_emb'][$pos_id] = $x;

        for ($li = 0; $li < $this->n_layer; $li++) {
            $x_in_att = $x;
            $x_norm = $this->rmsnorm($x, "pre_att_$li", $pos_id);

            $q = $this->linear_layer($x_norm, $this->params["layer{$li}.attn_wq"]);
            $k = $this->linear_layer($x_norm, $this->params["layer{$li}.attn_wk"]);
            $v = $this->linear_layer($x_norm, $this->params["layer{$li}.attn_wv"]);
            $keys[$li][]   = $k;
            $values[$li][] = $v;

            $attn_weights_cache = [];
            $x_attn = array_fill(0, $this->n_embd, 0.0);

            for ($h = 0; $h < $this->n_head; $h++) {
                $hs = $h * $this->head_dim;
                $q_h = array_slice($q, $hs, $this->head_dim);

                $attn_logits = [];
                foreach ($keys[$li] as $ki) {
                    $k_h = array_slice($ki, $hs, $this->head_dim);
                    $sum = 0.0;
                    for ($j = 0; $j < $this->head_dim; $j++)
                        $sum += $q_h[$j] * $k_h[$j];
                    $attn_logits[] = $sum / sqrt($this->head_dim);
                }

                $attn_weights = $this->softmax($attn_logits);
                $is_inference or $attn_weights_cache[$h] = $attn_weights;

                $head_out = array_fill(0, $this->head_dim, 0.0);
                foreach ($values[$li] as $t => $vi) {
                    $v_h = array_slice($vi, $hs, $this->head_dim);
                    $w = $attn_weights[$t];
                    for ($j = 0; $j < $this->head_dim; $j++)
                        $head_out[$j] += $w * $v_h[$j];
                }
                for ($j = 0; $j < $this->head_dim; $j++)
                    $x_attn[$hs + $j] = $head_out[$j];
            }

            $x_proj = $this->linear_layer($x_attn, $this->params["layer{$li}.attn_wo"]);
            for ($i = 0; $i < $this->n_embd; $i++)
                $x[$i] = $x_in_att[$i] + $x_proj[$i];

            $x_in_mlp    = $x;
            $x_norm_mlp  = $this->rmsnorm($x, "pre_mlp_$li", $pos_id);
            $h_fc        = $this->linear_layer($x_norm_mlp, $this->params["layer{$li}.mlp_fc1"]);

            $fc1_dim = count($h_fc);
            $h_act   = array_fill(0, $fc1_dim, 0.0);

            for ($i = 0; $i < $fc1_dim; $i++) {
                $val = $h_fc[$i];
                // ✅ ИСПРАВЛЕНИЕ: кэшируем $val ДО активации (нужно для backward)
                $is_inference or $this->cache["relu_{$li}_{$pos_id}"][$i] = $val;
                // ✅ ИСПРАВЛЕНИЕ: обычный ReLU (не relu²)
                $h_act[$i] = $val > 0 ? $val : 0.0;
            }

            $h_proj = $this->linear_layer($h_act, $this->params["layer{$li}.mlp_fc2"]);

            for ($i = 0; $i < $this->n_embd; $i++)
                $x[$i] = $x_in_mlp[$i] + $h_proj[$i];

            if ($is_inference)
                continue;

            $this->cache["x_norm_attn_{$li}_{$pos_id}"] = $x_norm;
            $this->cache["q_{$li}_{$pos_id}"]           = $q;
            $this->cache["attn_w_{$li}_{$pos_id}"]      = $attn_weights_cache;
            // ✅ ИСПРАВЛЕНИЕ: кэшируем h_act (после ReLU) — нужно для accumulate_grad mlp_fc2
            $this->cache["mlp_h_{$li}_{$pos_id}"]       = $h_act;
            $this->cache["x_norm_mlp_{$li}_{$pos_id}"]  = $x_norm_mlp;
            $this->cache["attn_out_{$li}_{$pos_id}"]    = $x_attn;
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
    function check_grad($param_name, $row, $col, $tokens, $eps = 1e-4) {
        // Forward с p+eps
        $this->params[$param_name][$row][$col] += $eps;
        $loss_plus = $this->train_one_doc_no_grad($tokens);
        $this->params[$param_name][$row][$col] -= 2 * $eps;
        $loss_minus = $this->train_one_doc_no_grad($tokens);
        $this->params[$param_name][$row][$col] += $eps;
        $numerical = ($loss_plus - $loss_minus) / (2 * $eps);
        $analytical = $this->grads[$param_name][$row][$col];
        echo "Diff: " . abs($numerical - $analytical) . "\n";  // должно быть < 1e-5
    }
     // ============================================================
     // ИСПРАВЛЕНИЕ 2: метод backward_layer()
     // Изменения:
     //   - градиент по входу для wk и wv накапливается по ВСЕМ шагам t
     //   - убран неправильный блок "только для $limit - 1"
     // ============================================================
    protected function backward_layer(array $d_x, int $li, int $pos_id, array &$keys, array &$values): array {
        // --- MLP backward ---
        $this->accumulate_grad("layer{$li}.mlp_fc2", $d_x, $this->cache["mlp_h_{$li}_{$pos_id}"]);
        $d_h = $this->backward_linear_input($d_x, $this->params["layer{$li}.mlp_fc2"]);

        $relu_cache = $this->cache["relu_{$li}_{$pos_id}"];
        $d_fc1_out = [];
        for ($i = 0, $cnt = count($d_h); $i < $cnt; $i++) {
            // ✅ ИСПРАВЛЕНИЕ: производная ReLU = 1 если x>0, иначе 0
            //    relu_cache содержит x ДО активации
            $d_fc1_out[$i] = $d_h[$i] * ($relu_cache[$i] > 0 ? 1.0 : 0.0);
        }

        $this->accumulate_grad("layer{$li}.mlp_fc1", $d_fc1_out, $this->cache["x_norm_mlp_{$li}_{$pos_id}"]);
        $d_x_norm_mlp   = $this->backward_linear_input($d_fc1_out, $this->params["layer{$li}.mlp_fc1"]);
        $d_x_mlp_input  = $this->backward_rmsnorm($d_x_norm_mlp, "pre_mlp_$li", $pos_id);

        // Residual MLP
        $d_input = [];
        for ($i = 0; $i < $this->n_embd; $i++)
            $d_input[$i] = $d_x_mlp_input[$i] + $d_x[$i];

        // --- Attention backward ---
        $this->accumulate_grad("layer{$li}.attn_wo", $d_input, $this->cache["attn_out_{$li}_{$pos_id}"]);
        $d_attn_ctx = $this->backward_linear_input($d_input, $this->params["layer{$li}.attn_wo"]);

        $d_q_total          = array_fill(0, $this->n_embd, 0.0);
        $attn_weights_cache = $this->cache["attn_w_{$li}_{$pos_id}"];
        $limit              = count($attn_weights_cache[0]);

        // ✅ ИСПРАВЛЕНИЕ: накапливаем градиенты по входу x для wk и wv
        //    по ВСЕМ временным шагам t, а не только по последнему
        $d_x_k_accum = array_fill(0, $this->n_embd, 0.0);
        $d_x_v_accum = array_fill(0, $this->n_embd, 0.0);

        // Сначала вычислим d_scores для всех голов и всех t (нужны ниже)
        $d_scores_all = []; // [head][t]

        for ($h = 0; $h < $this->n_head; $h++) {
            $hs         = $h * $this->head_dim;
            $d_head_out = array_slice($d_attn_ctx, $hs, $this->head_dim);
            $attn_w     = $attn_weights_cache[$h];

            // dL/d(attn_logits) до softmax
            $d_attn_logits = array_fill(0, $limit, 0.0);
            for ($t = 0; $t < $limit; $t++) {
                $v_h = array_slice($values[$li][$t], $hs, $this->head_dim);
                for ($j = 0; $j < $this->head_dim; $j++)
                    $d_attn_logits[$t] += $d_head_out[$j] * $v_h[$j];
            }

            // Jacobian softmax: d_scores = p * (d - sum(p*d))
            $sum_p_ds = 0.0;
            for ($t = 0; $t < $limit; $t++)
                $sum_p_ds += $attn_w[$t] * $d_attn_logits[$t];

            $d_scores = [];
            for ($t = 0; $t < $limit; $t++)
                $d_scores[$t] = $attn_w[$t] * ($d_attn_logits[$t] - $sum_p_ds);

            $d_scores_all[$h] = $d_scores;

            $q_curr = array_slice($this->cache["q_{$li}_{$pos_id}"], $hs, $this->head_dim);

            for ($t = 0; $t < $limit; $t++) {
                $k_t      = array_slice($keys[$li][$t], $hs, $this->head_dim);
                $score_t  = $d_scores[$t];
                $x_t      = $this->cache["x_norm_attn_{$li}_{$t}"];

                // Градиент по Q (только текущий pos_id)
                for ($j = 0; $j < $this->head_dim; $j++)
                    $d_q_total[$hs + $j] += $score_t * $this->scale_head * $k_t[$j];
                // Градиент весов WK
                for ($j = 0; $j < $this->head_dim; $j++) {
                    $grad_k   = $score_t * $this->scale_head * $q_curr[$j];
                    $row_idx  = $hs + $j;
                    foreach ($x_t as $col => $x_val)
                        $this->grads["layer{$li}.attn_wk"][$row_idx][$col] += $grad_k * $x_val;
                }

                // Градиент весов WV
                for ($j = 0; $j < $this->head_dim; $j++) {
                    $grad_v  = $attn_w[$t] * $d_head_out[$j];
                    $row_idx = $hs + $j;
                    foreach ($x_t as $col => $x_val)
                        $this->grads["layer{$li}.attn_wv"][$row_idx][$col] += $grad_v * $x_val;
                }

                // ✅ ИСПРАВЛЕНИЕ: градиент по ВХОДУ x для wk и wv
                //    накапливаем по всем t (раньше бралось только t = limit-1)
                $d_k_t = array_fill(0, $this->n_embd, 0.0);
                $d_v_t = array_fill(0, $this->n_embd, 0.0);
                for ($j = 0; $j < $this->head_dim; $j++) {
                    $d_k_t[$hs + $j] = $score_t * $this->scale_head * $q_curr[$j];
                    $d_v_t[$hs + $j] = $attn_w[$t] * $d_head_out[$j];
                }
                $dx_k_t = $this->backward_linear_input($d_k_t, $this->params["layer{$li}.attn_wk"]);
                $dx_v_t = $this->backward_linear_input($d_v_t, $this->params["layer{$li}.attn_wv"]);
                for ($i = 0; $i < $this->n_embd; $i++) {
                    $d_x_k_accum[$i] += $dx_k_t[$i];
                    $d_x_v_accum[$i] += $dx_v_t[$i];
                }
            }
        }

        $this->accumulate_grad("layer{$li}.attn_wq", $d_q_total, $this->cache["x_norm_attn_{$li}_{$pos_id}"]);
        $d_x_q = $this->backward_linear_input($d_q_total, $this->params["layer{$li}.attn_wq"]);

        // Суммируем градиенты от Q, K, V для нормализованного входа
        $d_x_norm_attn = [];
        for ($i = 0; $i < $this->n_embd; $i++)
            $d_x_norm_attn[$i] = $d_x_q[$i] + $d_x_k_accum[$i] + $d_x_v_accum[$i];

        $d_x_att_input = $this->backward_rmsnorm($d_x_norm_attn, "pre_att_$li", $pos_id);

        // Residual Attention
        for ($i = 0; $i < $this->n_embd; $i++)
            $d_input[$i] += $d_x_att_input[$i];

        return $d_input;
    }


    // Хелпер: накопление градиента для матрицы W (dW = dy * dx^T)
    protected function accumulate_grad(string $key, array $dy, array $x) {
        $w_rows = count($dy);
        $w_cols = count($x);
        for ($i = 0; $i < $w_rows; $i++) {
            for ($j = 0; $j < $w_cols; $j++)
                $this->grads[$key][$i][$j] += $dy[$i] * $x[$j];
        }
    }
    
    // Хелпер: расчет градиента на вход Linear слоя (dx = dy * W^T)
    protected function backward_linear_input(array $dy, array $w) {
        $dx = array_fill(0, $w_cols = count($w[0]), 0.0);
        $w_rows = count($w);
        for ($i = 0; $i < $w_rows; $i++) { // rows of W (out features)
            for ($j = 0; $j < $w_cols; $j++) // cols of W (in features)
                $dx[$j] += $dy[$i] * $w[$i][$j];
        }
        return $dx;
    }

    function linear_layer($x, $w) {
        $out = [];
        foreach ($w as $row) {
            $sum = 0.0;
            foreach ($row as $i => $wi)
                $sum += $wi * $x[$i];
            $out[] = $sum;
        }
        return $out;
    }

    function softmax($logits) {
        $max_val = max($logits);
        $exps = array_map(fn($val) => exp($val - $max_val), $logits);
        $total = array_sum($exps);
        return array_map(fn($e) => $e / $total, $exps);
    }

    // ============================================================
    // ИСПРАВЛЕНИЕ 4: метод backward_rmsnorm()
    // Уточнение: dot вычисляется через d_z (уже умноженный на gamma),
    // что точно соответствует аналитической производной RMSNorm.
    // Формула: grad_x_i = scale * (d_z_i - x_i * scale² * dot(x, d_z) / n)
    // ============================================================
    function backward_rmsnorm(array $grad_output, string $param_name, int $pos_id): array {
        [$x, $scale, $n] = $this->cache["{$param_name}_{$pos_id}"];
        $gamma = $this->params[$param_name];
        $d_z = $grad_input = [];
        $dot = 0.0;
        for ($i = 0; $i < $n; $i++) {
            $this->grads[$param_name][$i] += $grad_output[$i] * $x[$i] * $scale;
            $d_z[$i] = $grad_output[$i] * $gamma[$i];
            $dot += $x[$i] * $d_z[$i];
        }
        $coeff = ($dot * $scale * $scale) / $n;
        for ($i = 0; $i < $n; $i++) {
            $grad_input[$i] = $d_z[$i] * $scale - $x[$i] * $coeff;
        }
        return $grad_input;
    }

    // $param_name - weights name ("init_norm", "pre_att_$li", "pre_mlp_$li")
    function rmsnorm($x, $param_name, $pos_id) {
        $ms = 0.0;
        $n = count($x);
        foreach ($x as $xi)
            $ms += $xi * $xi;
        $ms /= $n;
        $scale = 1.0 / sqrt($ms + 1e-5);

        $this->cache["{$param_name}_{$pos_id}"] = [$x, $scale, $n];

        $gamma = $this->params[$param_name];
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $out[] = $x[$i] * $scale * $gamma[$i];
        }
        return $out;
    }
}
