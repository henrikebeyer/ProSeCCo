import pandas as pd
import numpy as np
from statistics import mean
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

ty = "props" # "locutions" "props"

accs = []
f1s = []
for i in range(1,4):
    pred_file = f"~/Dokumente/SelfContra/baselines/SemBERT_eval/_pred_results_{ty}_{i}.tsv"
    gold_file = "~/Dokumente/SelfContra/SelfContra_corpus/SelfContra_final.tsv"

    pred_df = pd.read_csv(pred_file, sep="\t")
    gold_df = pd.read_csv(gold_file, sep="\t")

    """
    #gold_labs = list(gold_df["label"])
    preds = list(pred_df["prediction"])

    correct = 0
    correct_pred = []

    for gold_lab, pred in zip(gold_labs, preds):
        if gold_lab != 1 and pred == "entailment":
            correct += 1
            correct_pred.append(True)
        elif gold_lab != 1 and pred == "neutral":
            correct += 1
            correct_pred.append(True)
        elif gold_lab == 1 and pred == "contradiction":
            correct += 1
            correct_pred.append(True)
        else:
            correct_pred.append(False)

    print(correct)
    """

    #comparison_df = pd.DataFrame()
    #comparison_df["ID"] = gold_df["ID"]
    #comparison_df["locution1"] = gold_df["locution1"]
    #comparison_df["locution2"] = gold_df["locution2"]
    #comparison_df["gold_label"] = gold_df["label"]
    #comparison_df["prediction"] = pred_df["prediction"]
    #comparison_df["correct"] = correct_pred

    #print(comparison_df["correct"].value_counts())

    gold = list(gold_df["label"])

    pred = [1 if lab == "contradiction" else 0 for lab in pred_df["prediction"]]

    acc = accuracy_score(gold, pred)
    f1 = f1_score(gold, pred)
    
    accs.append(acc)
    f1s.append(f1)


print("f1:", mean(f1s), "+-", np.std(np.asarray(f1s)))
print("acc:", mean(accs), "+-", np.std(np.asarray(accs)))


#comparison_df.to_csv("Compared_eval_SemBERT_locutions.tsv", sep="\t", index=False)

#print(gold_df["prep_needed"].value_counts())

