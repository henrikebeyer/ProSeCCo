import pandas as pd
import os
import sys
import shutil
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F

# ========== CONFIG ==========
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL = sys.argv[1]  # e.g., "facebook/bart-large-mnli"
model_name = MODEL.replace("/","_")
DATASET_FILE = "/home/henrike/Self-Contradiction_project/data/decode_prosecco_size_df.tsv"
OUTPUT_DIR = f"/home/henrike/Self-Contradiction_project/comparative_study/"
OUTPUT_FILE = f"{OUTPUT_DIR}/predict_decode_{model_name}.tsv"
SAVE_EVERY = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)


print(DEVICE)

# ========== Load Data & Split ==========
df_full = pd.read_csv(DATASET_FILE, sep="\t")

print(f"Dataset: {len(df_full)}")

# ========== Dataset ==========
class NLIDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256, label_col="nli"):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = {l: i for i, l in enumerate(LABEL_MAPPING)}
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        inputs = self.tokenizer(
            str(row['seg1']),
            str(row['seg2']),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        if self.label_col in row:
            item['labels'] = torch.tensor(self.label2id[str(row[self.label_col]).lower()])
        return item

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(DEVICE)

LABEL_MAPPING = list(model.config.id2label.values())

if MODEL == "FacebookAI/roberta-large-mnli":
    LABEL_MAPPING = [label.lower() for label in LABEL_MAPPING]


# ========== Inference with Progress and Periodic Saving ==========
df_full = df_full.copy()
df_full[f'{model_name}_pred'] = pd.NA

# Add columns for probabilities
for label in LABEL_MAPPING:
    df_full[f'{model_name}_prob_{label}'] = pd.NA

dataset = NLIDataset(df_full, tokenizer, label_col=None)
dataloader = DataLoader(dataset, batch_size=32)

completed = 0
batch_num = 0

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Classifying"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).cpu().numpy()  # shape: (batch_size, num_labels)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        labels = [LABEL_MAPPING[p] for p in preds]
        start = batch_num * 32
        end = start + len(labels)
        df_full.iloc[start:end, df_full.columns.get_loc(f'{model_name}_pred')] = labels

        # Save probabilities for each label
        for i, label in enumerate(LABEL_MAPPING):
            df_full.iloc[start:end, df_full.columns.get_loc(f'{model_name}_prob_{label}')] = probs[:, i]

        completed += len(labels)
        batch_num += 1

        if batch_num % SAVE_EVERY == 0:
            # Save probabilities to a separate file
            prob_cols = [f'{model_name}_ft_prob_{label}' for label in LABEL_MAPPING]
            df_full.to_csv(OUTPUT_FILE, sep='\t', index=False)
            print(f"[Checkpoint] Saved after {completed} examples.")

# ========== Final Save ==========
df_full.to_csv(OUTPUT_FILE, sep='\t', index=False)
# Save probabilities to a separate file
print(f"✅ Final save complete. Total labeled: {completed}")