# pretraining-for-structured


## 📚 Citation

If you use this code or build upon this work, **please cite**:

```bibtex
@inproceedings{fadlon-bar-2026-much,
    title = "How Much Pretraining Does Structured Data Need?",
    author = "Fadlon, Daniel  and
      Bar, Kfir",
    editor = "Demberg, Vera  and
      Inui, Kentaro  and
      Marquez, Llu{\'i}s",
    booktitle = "Proceedings of the 19th Conference of the {E}uropean Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.eacl-long.154/",
    doi = "10.18653/v1/2026.eacl-long.154",
    pages = "3352--3365",
    ISBN = "979-8-89176-380-7"
}
```



## Methods

This section describes the experimental methods available in this project and how to execute them.

**Note:** experiments in this repo are highly controlled by the YAML configuration passed into the training script. Start by looking at the example configs in `configs/examples/`, then modify the fields you need (model, reinit strategy, train size, etc.).

---

### 1. Pretraining vs Full Reinitialization

Compare the effect of pretrained weights vs training from scratch using random re-initialization.

#### 1.a. Regular Finetuning

Standard finetuning (that preserves pretrained weights). The model starts with knowledge learned during pretraining.

```bash
python experiments/run_finetuning_script.py configs/examples/regular_finetuning.yaml output/pt_llama
```

**Key config settings:**
```yaml
pretrained_model_id: "meta-llama/Llama-3.2-1B-Instruct"
```

#### 1.b. Full Reinitialization

Reinitialize all weights with random values from a zero-mean normal distribution (same standard deviation as original), so training begins from scratch.

```bash
python experiments/run_finetuning_script.py configs/examples/fully_reinitialize.yaml output/fully_reinit_model
```

**Key config settings:**
```yaml
reinit_strategy: "full"  # Reinitialize ALL modules via model.apply()
```

---

### 2. Training Set Size

Analyze performance as a function of training data size.

#### 2.a. Execute Different Train Sizes

Control the amount of training data using either **a ratio** (`percentage`) or **an exact number of examples** (`n`).

This can be combined with any other training configuration (regular finetuning, full reinitialization, or layer reinitialization) by setting `train_set_size` in the YAML you already use.

**Config for percentage-based sizing:**
```yaml
train_set_size:
  percentage: 0.5  # Use 50% of training data
```

**Config for exact sample count:**
```yaml
train_set_size:
  n: 1000  # Use exactly 1000 samples
```

---

### 3. Layer-wise Analysis

Study the effect of **pretrained layers** by keeping a prefix (“head”) of the model pretrained, while reinitializing the remaining layers (“tail”) and finetuning.

#### 3.a. Reinitialize Layers from a Specified Layer to the End

Reinitialize transformer layers from a starting layer through the final layer (the “tail” of the model). Layers before `reinit_from_layer` remain pretrained (the “head”).

```bash
python experiments/run_finetuning_script.py configs/examples/reinit_layers.yaml output/reinit_layers_model
```

**Key config settings:**
```yaml
reinit_strategy: "layers"  # Reinitialize transformer layers only
reinit_from_layer: 10      # Reinitialize from layer 10 to the end
```

**How to think about layer numbering (for a model with `num_hidden_layers = N`):**
| `reinit_from_layer` | Layers Reinitialized | Layers Kept Pretrained |
|---------------------|----------------------|------------------------|
| 0                   | 0-(N-1) (all)        | None                   |
| k                   | k-(N-1)              | 0-(k-1)                |

Create multiple configs that are identical except for `reinit_from_layer` (e.g., `0`, a middle layer index, and `N-1`), then run each config:

```bash
python experiments/run_finetuning_script.py <path_to_config.yaml> <output_dir>
```

This allows analysis of how much the model relies on pretrained representations at different depths.

---

### 4. Truncated-Depth Finetuning (first layer only)

Train on **pretrained weights**, but structurally truncate the transformer to only the first \(N\) layers (instead of training the full depth).

An example config for **first-layer-only finetuning** is provided:
- `configs/examples/first_layer_finetuning.yaml`

Run it like any other finetuning config:

```bash
python experiments/run_finetuning_script.py configs/examples/first_layer_finetuning.yaml output/first_layer_only_model
```

**Key config setting:**

```yaml
use_first_n_layers: 1
```

---

### 5. Probing Experiments

This repo includes a probing workflow that:
- **Extracts embeddings** from a specified transformer layer for train/valid/test
- **Trains a linear classifier** ("linear probe", e.g. logistic regression) on those embeddings

The unified entrypoint is:

```bash
python experiments/run_probing.py --run-config configs/examples/probing_run.yaml
```

#### 4.a. Probing a specific layer

Layer selection is controlled by the `layer` field in the probing run-config YAML:

```yaml
command: pipeline
embed_config: configs/examples/probing_embed.yaml
layer: 15
model_dir: output/probing_models/example_run
```

This will probe the representation at **layer 15** by creating embeddings at that layer and then running classification.

- The **embedding-definition config** (`embed_config`, e.g. `configs/examples/probing_embed.yaml`) defines:
  - the model id/path (`model_path_or_id`)
  - parquet paths for train/valid/test
  - which column to embed (`column_to_embed`) and which label column to use (`label_column`)
- The **run-config** defines:
  - what to execute (`command: pipeline|embed|classify`)
  - which layer to probe (`layer`)
  - where to save/load the probe classifier (`model_dir`)
