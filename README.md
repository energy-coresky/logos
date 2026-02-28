
# DEV ware, GPT-AI sandbox

A PHP port of Andrej Karpathy's microGPT implementation for the Coresky DEV-ware sandbox

This ware provides a pure PHP implementation of a GPT-style Transformer model, designed to run as a plugin within the **Coresky Framework**. It allows users to train, save, and infer text generation models directly in the browser sandbox without external dependencies (like Python or C extensions).

The architecture is based on Andrej Karpathy's **micrograd** and **nanoGPT** projects, ported to PHP to demonstrate the core mechanics of autograd and transformers.

**Source Inspiration:** [karpathy/microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) |
[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) |
[karpathy/micrograd](https://github.com/karpathy/micrograd)

## Features

*   **Pure PHP Implementation**: No heavy dependencies, runs on standard PHP 8.0+.
*   **Autograd Engine**: Includes a `Value` class for automatic differentiation (backpropagation).
*   **Transformer Architecture**: Supports configurable layers, heads, and embedding sizes.
*   **Binary Serialization**: Custom `GPT_Pack` class supports FP32, INT8, and INT4 quantization for efficient model storage.
*   **Sandbox Ready**: Designed for the Coresky DEV-TOOLS environment.

## Quick Start (PHP API)

```php
// 1. Initialize with dataset
$docs = array_map('trim', file('names.txt'));
$gpt = new GPT_Run($docs);

// 2. Configure architecture
$gpt->n_layer = 2;
$gpt->n_embd  = 32;
$gpt->build_vocab($docs); // Re-calc params

// 3. Train
$gpt->init_weights();
$gpt->train($docs, $n_steps = 2000, $learning_rate = 0.0003);

// 4. Inference
foreach ($gpt->inference(0.6, 5) as $sample) {
    echo "Generated: " . $sample . "\n";
}

// 5. Save model (Quantized)
$settings = ['n_layer' => 2, 'n_embd' => 32, 'dataset_file' => 'names.txt'];
GPT_Pack::save('model.bin', $gpt->params, $settings, GPT_Pack::Q_INT8);
```

## Console Usage (CLI)

You can run training and inference directly from the terminal using the `sky` command.

**Syntax:**
```bash
>sky logos z param1=value1 param2=value2 ...
```

**Main Parameters:**
*   `txt=filename`: Load dataset from the `txt/` directory.
*   `bin=filename`: Load or save model to the `bin/` directory.
*   `qtz=4|8|32`: Quantization type for saving (INT4, INT8, FP32). If omitted, model won't be saved.
*   `rnd=seed`: Random seed (default: 42). Set `rnd=0` to disable.

**Short Parameters (Aliases):**
*   `x`: Number of training steps (`n_train`)
*   `y`: Number of inference samples (`n_inference`)
*   `e`: Embedding size (`n_embd`)
*   `l`: Number of layers (`n_layer`)
*   `r`: Learning rate
*   `t`: Temperature

**Examples:**

1.  **Train from scratch** (take dataset from `txt/math.txt`, train 500 steps):
    ```bash
    >sky logos z txt=math x=500 y=11
    ```

2.  **Inference only** (load model from `bin/math.bin`):
    ```bash
    >sky logos z bin=math y=11
    ```

3.  **Train and Save** (train 10000 steps, save as quantized INT8):
    ```bash
    >sky logos z txt=math bin=math x=10000 qtz=8
    ```

## To Do

-   [x] **Performance Core Rewrite**: Refactor the engine to use native PHP arrays instead of the `Value` object wrappers for a ~50x speed boost.
-   [ ] **UI Integration**: Add visual progress bars and Loss charts to the Coresky Sandbox interface.
-   [ ] **KV-Cache Optimization**: Optimize inference speed by caching key-value states.
-   [ ] **Advanced Regularization**: Implement Dropout and LayerNorm for better convergence on larger datasets.

## Status

**_Under development_**

