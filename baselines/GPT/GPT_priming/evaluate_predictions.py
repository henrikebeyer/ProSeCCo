import pandas as pd
import numpy as np
import csv
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from statistics import mean

seeds = [42, 121, 52, 78]
nums = [2, 8, 16, 32]
ty = "proposition" # "locution" "proposition"


accs =  [[],[],[],[],[]]
f1s = [[],[],[],[],[]]

for i in range(len(nums)):
    num = nums[i]
    for j in range(len(seeds)):
        seed = seeds[j]
        for run in range(1,4):

            pred_df = pd.read_csv(f"predictions/{ty}_predictions_{num}_{seed}_{run}.tsv", sep="\t")
            preds = [pred.lower().replace("**","").replace(".","").replace("_","").replace("'","").strip() for pred in list(pred_df["prediction"])]
            gold_labels = list(pred_df["gold_standard"])

            prime_df = pd.read_csv(f"~/Dokumente/SelfContra/baselines/GPT_priming/primes/{ty}_primes_{num}_{seed}.tsv", sep="\t")
            prime_labels = list(prime_df["label"])

            #print(set(preds))


            for i in range(len(preds)):
                pred = preds[i]
                if len(pred.split()) > 1 and "non-contradiction" in pred: #and pred.lower().replace("'", "").replace(".", "").endswith("non-contradiction"):
                    #print(pred)
                    preds[i] = "non-contradiction"
                    
                elif len(pred.split()) > 1 and (pred.replace("'", "").replace(".", "").endswith("self-contradiction") or pred.replace("'", "").replace(".", "").endswith(" contradiction")):
                    preds[i] = "self-contradiction"

                elif pred == "non-contradition":
                    preds[i] = "non-contradiction"

                elif "non-contradictory" in pred:
                    preds[i] = "non-contradiction"
                
                elif "self-contradictory" in pred:
                    preds[i] = "non-contradiction"
                elif pred == "`self-contradiction`":
                    preds[i] = "self-contradiction"

                elif "self-contradiction" in pred:
                    preds[i] = "self-contradiction"
                elif "no contradiction" in pred:
                    preds[i] = "non-contradiction"
                elif pred == "self-contradition":
                    preds[i] = "self-contradiction"
                elif "cannot classify" in pred:
                    preds[i] = "non-contradiction"

            #print(set(preds))

            pred_df["prediction"] = preds

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
            #print(acc)
            #print(f1)

            if num == 2:
                accs[0].append(acc)
                f1s[0].append(f1)
            elif num == 8:
                accs[1].append(acc)
                f1s[1].append(f1)
            elif num == 16:
                accs[2].append(acc)
                f1s[2].append(f1)
            elif num == 32:
                accs[3].append(acc)
                f1s[3].append(f1)

            
            with open(f"evaluations/{ty}_evaluation_{num}_{seed}_{run}", "w") as fout:
                fout.write(f"---Evaluation of run {run} of {ty} trial with {num} primes and random seed {seed} \n\n")
                fout.write(f"- self-contradiction primes: {prime_labels.count('self-contradiction')} \n")
                fout.write(f"- non-contradiction: {prime_labels.count('non-contradiction')}\n\n")

                fout.write(f"-accuracy: {acc} \n")
                fout.write(f"-F1-Score: {f1} \n\n")

                fout.write("---Confusion matrix--- \n")
                fout.write("\t \t pred pos \t pred neg \n")
                fout.write(f"act pos \t {conf_matrix[0][0]} \t\t {conf_matrix[0][1]} \n")
                fout.write(f"act neg \t {conf_matrix[1][0]} \t\t {conf_matrix[1][1]}")
            
            with open(f"evaluations/{ty}_evaluation_all", "a") as fout:
                fout.write(f"+++Evaluation of run {run} of {ty} trial with {num} primes and random seed {seed}+++ \n\n")
                fout.write(f"- self-contradiction primes: {prime_labels.count('self-contradiction')} \n")
                fout.write(f"- non-contradiction: {prime_labels.count('non-contradiction')}\n\n")

                fout.write(f"-accuracy: {acc} \n")
                fout.write(f"-F1-Score: {f1} \n\n")

                fout.write("---Confusion matrix--- \n")
                fout.write("\t \t pred pos \t pred neg \n")
                fout.write(f"act pos \t {conf_matrix[0][0]} \t\t {conf_matrix[0][1]} \n")
                fout.write(f"act neg \t {conf_matrix[1][0]} \t\t {conf_matrix[1][1]} \n\n")


trials = ["2-shot", "8-shot", "16-shot", "32-shot"]

#print(accs)
#print(f1s)

with open(f"mean_std_all.tsv", "w") as fout:
    writer = csv.writer(fout)
    writer.writerow(["run", "f1", "std", "acc", "std"])
    for i in range(len(trials)):
        trial = trials[i]
        f1_mean = mean(f1s[i])
        f1_std = np.std(np.asanyarray(f1s[i]))
        acc_mean = mean(accs[i])
        acc_std = np.std(np.asarray(accs[i]))
        writer.writerow([trial, f1_mean, f1_std, acc_mean, acc_std])

print("f1_2:", mean(f1s[0]))
print("f1_2_std:", np.std(np.asarray(f1s[0])))
print("acc_2:", mean(accs[0]))
print("acc_2_std:", np.std(np.asarray(accs[0])))
print("f1_8:", mean(f1s[1]))
print("f1_8_std:", np.std(np.asarray(f1s[1])))
print("acc_8:", mean(accs[1]))
print("acc_8_std:", np.std(np.asarray(accs[1])))
print("f1_16:", mean(f1s[2]))
print("f1_16_std:", np.std(np.asarray(f1s[2])))
print("acc_16:", mean(accs[2]))
print("acc_16_std:", np.std(np.asarray(accs[2])))
print("f1_32:", mean(f1s[3]))
print("f1_32_std:", np.std(np.asarray(f1s[3])))
print("acc_32:", mean(accs[3]))
print("acc_32_std:", np.std(np.asarray(accs[3])))
