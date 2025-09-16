import pandas as pd
import numpy as np
from statistics import mean
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

ty = "locution" # "locution" "proposition"

f1s = []
accs = []

for run in range(1,4):

    pred_df = pd.read_csv(f"predictions/GPT_zeroShot_{ty}_predictions_{run}.tsv", sep="\t")
    preds = [pred.lower().replace("'","").replace("**","").replace(".","").strip() for pred in list(pred_df["prediction"])]
    gold_labels = ["non-contradiction" if label==0 else "self-contradiction" for label in pred_df["gold_standard"]]
    pred_df["gold_label"] = gold_labels

#prime_df = pd.read_csv(f"primes_{num}_{seed}.tsv", sep="\t")
#prime_labels = list(prime_df["label"])

    for i in range(len(preds)):
        pred = preds[i]
        if len(pred.split()) > 1 and "non-contradiction" in pred: #and pred.lower().replace("'", "").replace(".", "").endswith("non-contradiction"):
            #print(pred)
            preds[i] = "non-contradiction"
            
        elif len(pred.split()) > 1 and pred.replace("'", "").replace(".", "").endswith("self-contradiction"):
            preds[i] = "self-contradiction"

        elif "self-contradiction" in pred:
            preds[i] = "self-contradiction"

        elif "non-contradictory" in pred:
            preds[i] = "non-contradiction"

        elif pred == "contradiction":
            preds[i] = "self-contradiction"

        elif pred.split()[-1] == "contradiction":
            preds[i] = "self-contradiction"

        elif "self-contradictory" in pred:
            preds[i] = "self-contradiction"

    pred_df["prediction"] = preds

    print(set(preds))
    correct = []

    for pred, label in zip(preds, gold_labels):
        if pred == label:
            correct.append(True)
        else:
            correct.append(False)

    pred_df["correct"] = correct


    acc = accuracy_score(gold_labels, preds)
    f1 = f1_score([1 if lab == "self-contradiction" else 0 for lab in gold_labels], [1 if lab == "self-contradiction" else 0 for lab in preds])
    conf_matrix = confusion_matrix(gold_labels, preds)
    accs.append(acc)
    f1s.append(f1)

print(f"---Evaluation of trial zero Shot on {ty}\n\n")
print(f"- accuracy: {mean(accs)} +- {np.std(np.asarray(accs))} \n")
print(f"- f1: {mean(f1s)} +-{np.std(np.asarray(f1s))} \n")
print(f"- confusion matrix: \n ")
print("\t \t pred pos \t pred neg \n")
print(f"act pos \t {conf_matrix[0][0]} \t\t {conf_matrix[0][1]} \n ")
print(f"act neg \t {conf_matrix[1][0]} \t\t {conf_matrix[1][1]} \n ")

#with open(f"evaluation_GPT_zeroShot_{ty}_2", "w") as fout:
#    fout.write(f"---Evaluation of zero shot trial \n\n")
    
#    fout.write(f"- accuracy: {acc} +- {np.std(np.asarray(accs))}\n")
#    fout.write(f"- F1: {f1} +- +-{np.std(np.asarray(f1s))}\n\n")

#    fout.write("---Confusion matrix---\n")
#    fout.write("\t \t pred pos \t pred neg \n")
#    fout.write(f"act pos \t {conf_matrix[0][0]} \t\t {conf_matrix[0][1]} \n ")
#    fout.write(f"act neg \t {conf_matrix[1][0]} \t\t {conf_matrix[1][1]} \n ")
#print(pred_df)
#pred_df.to_csv(f"predictions_{num}_{seed}_clean_eval.tsv", sep="\t", index=False)
