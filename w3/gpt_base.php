<?php

class GPT_Base
{
    // Architecture Params
    public int $n_embd = 16;
    public int $n_head = 4;
    public int $n_layer = 1;
    public int $block_size = 8;
    public int $head_dim;

    // Model State
    public array $params = [];
    public int $n_params = 0;
    public int $vocab_size = 0;

    // Tokenizer
    public array $stoi = [];
    public array $itos = [];
    public int $BOS = 0;

    // Define the model architecture
    public function gpt($token_id, $pos_id, &$keys, &$values) {
        $tok_emb = $this->params['wte'][$token_id];
        $pos_emb = $this->params['wpe'][$pos_id];
        
        $x = [];
        for ($i = 0; $i < count($tok_emb); $i++) {
            $x[] = $tok_emb[$i]->__add($pos_emb[$i]);
        }
        $x = rmsnorm_func($x);
        
        for ($li = 0; $li < $this->n_layer; $li++) {
            // Multi-head attention block
            $x_residual = $x;
            $x = rmsnorm_func($x);
            $q = linear_layer($x, $this->params["layer{$li}.attn_wq"]);
            $k = linear_layer($x, $this->params["layer{$li}.attn_wk"]);
            $v = linear_layer($x, $this->params["layer{$li}.attn_wv"]);
            
            $keys[$li][] = $k;
            $values[$li][] = $v;
            
            $x_attn = [];
            for ($h = 0; $h < $this->n_head; $h++) {
                $hs = $h * $this->head_dim;
                $q_h = array_slice($q, $hs, $this->head_dim);
                
                $k_h = [];
                foreach ($keys[$li] as $ki) {
                    $k_h[] = array_slice($ki, $hs, $this->head_dim);
                }
                
                $v_h = [];
                foreach ($values[$li] as $vi) {
                    $v_h[] = array_slice($vi, $hs, $this->head_dim);
                }
                
                $attn_logits = [];
                for ($t = 0; $t < count($k_h); $t++) {
                    $sum = new Value(0);
                    for ($j = 0; $j < $this->head_dim; $j++) {
                        $sum = $sum->__add($q_h[$j]->__mul($k_h[$t][$j]));
                    }
                    $attn_logits[] = $sum->__div(sqrt($this->head_dim));
                }
                
                $attn_weights = softmax($attn_logits);
                
                for ($j = 0; $j < $this->head_dim; $j++) {
                    $head_out_j = new Value(0);
                    for ($t = 0; $t < count($v_h); $t++) {
                        $head_out_j = $head_out_j->__add($attn_weights[$t]->__mul($v_h[$t][$j]));
                    }
                    $x_attn[] = $head_out_j;
                }
            }
            
            $x = linear_layer($x_attn, $this->params["layer{$li}.attn_wo"]);
            for ($i = 0; $i < count($x); $i++) {
                $x[$i] = $x[$i]->__add($x_residual[$i]);
            }
            
            // MLP block
            $x_residual = $x;
            $x = rmsnorm_func($x);
            $x = linear_layer($x, $this->params["layer{$li}.mlp_fc1"]);
            $x = array_map(fn($xi) => $xi->relu()->__pow(2), $x);
            $x = linear_layer($x, $this->params["layer{$li}.mlp_fc2"]);
            for ($i = 0; $i < count($x); $i++) {
                $x[$i] = $x[$i]->__add($x_residual[$i]);
            }
        }
        
        return linear_layer($x, $this->params['lm_head']);
    }
}

// ============================================================================
// Value class: Autograd implementation
// ============================================================================

class Value {
    public $data;
    public $grad;
    public $_backward;
    public $_prev;
    public $_op;

    public function __construct($data, $children = [], $op = '') {
        $this->data = $data;
        $this->grad = 0;
        $this->_backward = fn() => null;
        $this->_prev = $children;
        $this->_op = $op;
    }

    public function __add($other) {
        $other = $other instanceof Value ? $other : new Value($other);
        $out = new Value($this->data + $other->data, [$this, $other], '+');
        $out->_backward = function() use ($other, $out) {
            $this->grad += $out->grad;
            $other->grad += $out->grad;
        };
        return $out;
    }

    public function __mul($other) {
        $other = $other instanceof Value ? $other : new Value($other);
        $out = new Value($this->data * $other->data, [$this, $other], '*');
        $out->_backward = function() use ($other, $out) {
            $this->grad += $other->data * $out->grad;
            $other->grad += $this->data * $out->grad;
        };
        return $out;
    }

    public function __pow($other) {
        if (!is_int($other) && !is_float($other))
            throw new Exception("Only supporting int/float powers for now");
        $out = new Value(pow($this->data, $other), [$this], "**{$other}");
        $out->_backward = function() use ($other, $out) {
            $this->grad += ($other * pow($this->data, $other - 1)) * $out->grad;
        };
        return $out;
    }

    public function log_val() {
        $out = new Value(log($this->data), [$this], 'log');
        $out->_backward = function() use ($out) {
            $this->grad += (1 / $this->data) * $out->grad;
        };
        return $out;
    }

    public function exp_val() {
        $out = new Value(exp($this->data), [$this], 'exp');
        $out->_backward = function() use ($out) {
            $this->grad += $out->data * $out->grad;
        };
        return $out;
    }

    public function relu() {
        $out = new Value($this->data < 0 ? 0 : $this->data, [$this], 'ReLU');
        $out->_backward = function() use ($out) {
            $this->grad += ($out->data > 0 ? 1 : 0) * $out->grad;
        };
        return $out;
    }

    public function backward() {
        $topo = $visited = [];
        $build_topo = function($v) use (&$build_topo, &$topo, &$visited) {
            $v_id = spl_object_id($v);
            if (!isset($visited[$v_id])) {
                $visited[$v_id] = true;
                foreach ($v->_prev as $child)
                    $build_topo($child);
                $topo[] = $v;
            }
        };
        $build_topo($this);
        $this->grad = 1;
        foreach (array_reverse($topo) as $v)
            ($v->_backward)();
    }

    public function __neg() {
        return $this->__mul(-1);
    }

    public function __sub($other) {
        $other = $other instanceof Value ? $other : new Value($other);
        return $this->__add($other->__neg());
    }

    public function __div($other) {
        $other = $other instanceof Value ? $other : new Value($other);
        return $this->__mul($other->__pow(-1));
    }

    public function __toString() {
        return "Value(data={$this->data}, grad={$this->grad})";
    }
}
