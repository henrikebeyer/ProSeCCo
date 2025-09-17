import pandas as pd
import os
import sys
from numpy import mean, std
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score

MODEL = sys.argv[1]
PRED_DIR = f"/home/henrike/Self-Contradiction_project/comparative_study/{MODEL}_preds"

def read_in_fewShot_preds(PRED_DIR, k, seed, run_id):
    df = pd.read_csv(f"{PRED_DIR}/{MODEL}_preds_decode_{k}-Shot_{seed}_{run_id}.tsv", sep="\t")
    return df

def read_in_0Shot_preds(PRED_DIR, k, run_id):
    df = pd.read_csv(f"{PRED_DIR}/{MODEL}_preds_decode_{k}-Shot_{run_id}.tsv", sep="\t")
    return df

def evaluate_preds(pred_df, model):
    gold = pred_df["label"]
    preds = pd.Series([pred.lower().replace(".","") for pred in  pred_df[f"{model}_preds"]])

    labels = list(preds.unique())
    print(labels)
    
    
    f1 = f1_score(gold, preds, average="macro", labels=labels)
    acc = accuracy_score(gold, preds)
    precision = precision_score(gold, preds, average="macro", zero_division=0.0, labels=labels)
    recall = recall_score(gold, preds, average="macro", zero_division=0.0, labels=labels)

    return acc, f1, precision, recall

def evaluate_fewShot_runs(PRED_DIR, MODEL):
    
    for k in [2, 4, 8, 16, 32]:
        f1s = []
        accs = []
        precs = []
        recs = [] 

        for seed in [42, 52, 121, 78]:
            for run_id in range(1, 4):
                pred_df = read_in_fewShot_preds(PRED_DIR, k, seed, run_id)
                acc, f1, precision, recall = evaluate_preds(pred_df, MODEL)
                f1s.append(f1)
                accs.append(acc)
                precs.append(precision)
                recs.append(recall)

        print(len(f1s))
        print(f1s)
        print(f"Scores for {MODEL} {k}-Shot: \nF1:{mean(f1s):.4} std: {std(f1s):.4} \nAcc: {mean(accs):.4} std:{std(accs):.4} \nPrecision: {mean(precs):.4} std: {std(precs):.4} \nRecall: {mean(recs):.4} std: {std(recs):.4}")


def evaluate_0Shot_runs(PRED_DIR, MODEL):
    f1s = []
    accs = []
    precs = []
    recs = [] 

    for run_id in range(1, 4):
        pred_df = read_in_0Shot_preds(PRED_DIR, 0, run_id)
        acc, f1, precision, recall = evaluate_preds(pred_df, MODEL)
        f1s.append(f1)
        accs.append(acc)
        precs.append(precision)
        recs.append(recall)
    print(len(f1s))
    print(f1s)

    print(f"Scores for {MODEL} 0-Shot: \nF1:{mean(f1s):.4} std: {std(f1s):.4} \nAcc: {mean(accs):.4} std:{std(accs):.4} \nPrecision: {mean(precs):.4} std: {std(precs):.4} \nRecall: {mean(recs):.4} std: {std(recs):.4}")

evaluate_fewShot_runs(PRED_DIR, MODEL)
evaluate_0Shot_runs(PRED_DIR, MODEL)