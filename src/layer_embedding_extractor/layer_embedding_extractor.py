from concurrent.futures import ThreadPoolExecutor
from typing import List
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import numpy as np
from tqdm import tqdm

class LayerEmbeddingExtractor:
    def __init__(self, model_path_or_id: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model_path_or_id = model_path_or_id
        self.pretrained_model = AutoModelForCausalLM.from_pretrained(model_path_or_id,
                                                    torch_dtype=torch.float16,
                                                    device_map="auto",
                                                    attn_implementation="eager",
                                                    use_safetensors=True)

        self.pretrained_model.config.output_hidden_states = True
        self.pretrained_model.config.return_dict = True

        self.expected_number_of_layers = self.pretrained_model.config.num_hidden_layers + 1

        self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'

    def extract_at_layer(self, texts: List[str], layer_num: int, batch_size: int = 8):
        all_chunks = []
        self.pretrained_model.eval()

        for i in tqdm(range(0, len(texts), batch_size), desc="Processing"):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1200).to(self.device)
            with torch.no_grad():
                outputs = self.pretrained_model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)

            all_hs = outputs.hidden_states  # tuple: [embeddings, layer1, ..., layerN]
            if all_hs is None:
                raise RuntimeError("hidden_states is None. Ensure output_hidden_states=True.")

            if len(all_hs) != self.expected_number_of_layers:
                raise RuntimeError(f"Invalid number of hidden states. Expected: {self.expected_number_of_layers}. but got {len(all_hs)}")

            # Validate layer index
            if not (-len(all_hs) <= layer_num < len(all_hs)):
                raise IndexError(
                    f"layer_num={layer_num} out of range for hidden_states of length {len(all_hs)} "
                    "(0 is token embeddings; last is final layer)."
                )

            layer = all_hs[layer_num] # [B, T_hs, H]
            attn_mask = inputs["attention_mask"]  # [B, T_mask]

            # Align lengths if they differ (e.g., BOS added)
            if attn_mask.size(1) != layer.size(1):
                diff = layer.size(1) - attn_mask.size(1)
                if diff > 0:
                    # model produced longer sequence than mask → pad mask with 1s where the extra token(s) are real
                    # If BOS was added, it's typically at the **left** for causal models.
                    attn_mask = torch.nn.functional.pad(attn_mask, (diff, 0), value=1)  # pad left
                else:
                    # mask longer than hidden states → truncate mask
                    attn_mask = attn_mask[:, :layer.size(1)]

            # Masked mean over sequence length (don’t average padding)
            attn_mask = attn_mask.unsqueeze(-1)   # [B, T, 1]
            masked = layer * attn_mask            # [B, T, H]
            lengths = attn_mask.sum(dim=1).clamp(min=1)  # [B, 1]
            pooled = masked.sum(dim=1) / lengths  # [B, H]

            all_chunks.append(pooled.to("cpu"))

        return torch.cat(all_chunks, dim=0).numpy()


    def build_embeddings_dataset_at_layer(self,
                layer_num: int,
                df: pd.DataFrame,
                set_type: str,
                out_dir: str,
                should_save_labels: bool,
                column="text",
                label_column="label"
                ):
        text_data = list(df[column])
        if should_save_labels:
            print(f"SAVE_LABELS to {out_dir}")
            self.save_labels_separately(df, out_dir, set_type, label_column)

        embeddings = self.extract_at_layer(text_data, layer_num)
        np.save(f"{out_dir}/X_{set_type}.npy", embeddings)

        print(f"Operation completed")


    @staticmethod
    def save_labels_separately(df: pd.DataFrame, out_dir: str, set_type: str, label_column: str):
        labels = df[label_column].to_numpy()
        np.save(f"{out_dir}/y_{set_type}.npy", labels)
