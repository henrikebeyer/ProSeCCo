import pandas as pd
import os
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score

PRED_FOLDER = "/home/henrike/Self-Contradiction_project/comparative_study/PLM_preds"
MODELS = ["facebook_bart-large-mnli", "FacebookAI_roberta-large-mnli", "sentence-transformers_nli-bert-base"]

def read_in_preds(pred_dir, model):
    df = pd.read_csv(f"{PRED_FOLDER}/predict_decode_{model}.tsv", sep="\t")
    return df

def evaluate_preds(pred_df, model):
    gold = pred_df["label"]
    preds = [1 if pred=="contradiction" else 0 for pred in pred_df[f"{model}_pred"]]
    
    f1 = f1_score(gold, preds, average="macro")
    acc = accuracy_score(gold, preds)
    precision = precision_score(gold, preds, average="macro", zero_division="warn")
    recall = recall_score(gold, preds, average="macro", zero_division="warn")

    print(f"Results for {model}: \nAccruacy: {acc:.4f} \nF1: {f1:.4f} \nPrecision: {precision:.4f}, \nRecall: {recall:.4f}")

pred_df = read_in_preds(PRED_FOLDER, MODELS[2])

evaluate_preds(pred_df, MODELS[2])