<?php

class GPT_Run extends GPT_Base
{
    // Optimizer State (Adam)
    private array $m = [];
    private array $v = [];
    private float $beta1 = 0.9;
    private float $beta2 = 0.95;
    private float $eps_adam = 1e-8;

    public function __construct(array $docs = [])
    {
        $this->head_dim = intval($this->n_embd / $this->n_head);
        if (!empty($docs)) {
            $this->build_vocab($docs);
        }
    }

    public function build_vocab(array $docs): void
    {
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
    }

    public function load(string $filename): void
    {
        if (!file_exists($filename))
            throw new Error("File not found: $filename");
        
        $this->params = GPT_Pack::decode(file_get_contents($filename));
        array_walk_recursive($this->params, fn(&$v) => $v = new Value($v));
        $this->m = $this->v = array_fill(0, $this->n_params, 0.0);
    }

    public function save(string $filename, int $quantization = GPT_Pack::Q_FP32): void
    {
        $weights = $this->params;
        array_walk_recursive($weights, fn(&$v) => $v = $v->data);
        file_put_contents($filename, GPT_Pack::encode($weights, $quantization));
    }

    /**
     * Инициализация случайных весов (Xavier/Normal)
     */
    public function init_weights(): void
    {
        $this->params = [
            'wte' => matrix($this->vocab_size, $this->n_embd),
            'wpe' => matrix($this->block_size, $this->n_embd),
            'lm_head' => matrix($this->vocab_size, $this->n_embd)
        ];

        for ($i = 0; $i < $this->n_layer; $i++) {
            $this->params["layer{$i}.attn_wq"] = matrix($this->n_embd, $this->n_embd);
            $this->params["layer{$i}.attn_wk"] = matrix($this->n_embd, $this->n_embd);
            $this->params["layer{$i}.attn_wv"] = matrix($this->n_embd, $this->n_embd);
            $this->params["layer{$i}.attn_wo"] = matrix($this->n_embd, $this->n_embd, 0);
            $this->params["layer{$i}.mlp_fc1"] = matrix(4 * $this->n_embd, $this->n_embd);
            $this->params["layer{$i}.mlp_fc2"] = matrix($this->n_embd, 4 * $this->n_embd, 0);
        }

        $this->m = $this->v = array_fill(0, $this->n_params, 0.0);
    }

    public function train(array $docs, int $n_steps, float $learning_rate): void
    {
        $this->init_weights();
        $num_docs = count($docs);
        for ($step = 0; $step < $n_steps; $step++) {
            $doc = $docs[$step % $num_docs];
            $tokens = [$this->BOS];
            for ($i = 0; $i < strlen($doc); $i++) {
                $tokens[] = $this->stoi[$doc[$i]];
            }
            $tokens[] = $this->BOS;
            $n = min($this->block_size, count($tokens) - 1);

            // Forward pass
            $keys = $values = array_fill(0, $this->n_layer, []);
            $losses = [];

            for ($pos_id = 0; $pos_id < $n; $pos_id++) {
                $token_id = $tokens[$pos_id];
                $target_id = $tokens[$pos_id + 1];
                $probs = softmax(gpt($token_id, $pos_id, $keys, $values));
                $losses[] = $probs[$target_id]->log_val()->__neg();
            }

            // Backward pass
            $loss = new Value(0);
            foreach ($losses as $l)
                $loss = $loss->__add($l);
            $loss = $loss->__div($n);
            $loss->backward();

            // Adam update
            $lr_t = $learning_rate * (1 - $step / $n_steps); // Linear decay
            $i = 0;
            array_walk_recursive($this->params, function($p) use (&$i, $lr_t, $step) {
                $this->m[$i] = $this->beta1 * $this->m[$i] + (1 - $this->beta1) * $p->grad;
                $this->v[$i] = $this->beta2 * $this->v[$i] + (1 - $this->beta2) * ($p->grad ** 2);
                $m_hat = $this->m[$i] / (1 - pow($this->beta1, $step));
                $v_hat = $this->v[$i] / (1 - pow($this->beta2, $step));
                $p->data -= $lr_t * $m_hat / (sqrt($v_hat) + $this->eps_adam);
                $p->grad = 0;
                $i++;
            });

            printf("\rstep %4d / %4d | loss %.4f", $step + 1, $n_steps, $loss->data);
        }
        echo "\n";
    }

    public function inference(float $temperature = 0.6, int $count = 10): Generator
    {
        for ($i = 0; $i < $count; ) {
            $keys = $values = array_fill(0, $this->n_layer, []);
            $token_id = $this->BOS;
            $out = '';

            for ($pos_id = 0; $pos_id < $this->block_size; $pos_id++) {
                $logits = gpt($token_id, $pos_id, $keys, $values);
                $scaled_logits = array_map(fn($l) => $l->__div($temperature), $logits);
                $weights = array_map(fn($p) => $p->data, softmax($scaled_logits));
                $token_id = random_choices(range(0, $this->vocab_size - 1), $weights);

                if ($token_id == $this->BOS)
                    break;
                $out .= $this->itos[$token_id];
            }
            yield ++$i => $out;
        }
    }
}
