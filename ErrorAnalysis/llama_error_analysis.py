import pandas as pd
from statistics import mean, median
import numpy as np


ty = "locution"
nums = [2, 4, 8, 16, 32]
seeds = [42, 52, 78, 121]

for ty in ["locution", "proposition"]:
    FPs = []
    FNs = []
    perc_FP = []
    perc_FN = []
    for run in range(1, 4):
        pred_path = f"/home/oenni/Dokumente/Self-ContradictionProject/Additional_Material_Coling/baselines/Llama3.3-70B/0-shot/{ty}/Llama3.3-70B_{ty}_0-shot_{run}.tsv"
        pred_df = pd.read_csv(pred_path, sep="\t")

        preds = [pred.lower().replace("**","").replace(".","").replace("_","").replace("'","").strip() for pred in list(pred_df["prediction"])]

                #print(set(preds))
                
        for i in range(len(preds)):
            pred = preds[i]
            if "self-contr" in pred:
                preds[i] = "self-contradiction"
            elif "non-contr" in pred:
                preds[i] = "non-contradiction"

        #print(set(preds))

        loc_status = []

        pred_df["prediction"] = preds

        for gold, pred in zip(pred_df["gold_label"], preds):
            #print(pred, type(pred))
            if gold == "self-contradiction" and pred == "self-contradiction":
                loc_status.append("TP")
            elif gold == "non-contradiction" and pred == "non-contradiction":
                loc_status.append("TN")
            elif gold == "self-contradiction" and pred == "non-contradiction":
                loc_status.append("FN")
            elif gold == "non-contradiction" and pred == "self-contradiction":
                loc_status.append("FP")

        pred_df["eval_status"] = loc_status

        preds_FP = pred_df[pred_df["eval_status"] == "FP"]
        preds_FN = pred_df[pred_df["eval_status"] == "FN"]

        preds_FN.to_csv(f"/home/oenni/Dokumente/Self-ContradictionProject/Additional_Material_Coling/baselines/Llama3.3-70B/erroranalysis/Llama3.3-70B_FN_{ty}_{num}_{run}.tsv", sep="\t", index=False)
        preds_FP.to_csv(f"/home/oenni/Dokumente/Self-ContradictionProject/Additional_Material_Coling/baselines/Llama3.3-70B/erroranalysis/Llama3.3-70B_FP_{ty}_{num}_{run}.tsv", sep="\t", index=False)

    FP = pred_df["eval_status"].value_counts()["FP"]
    FN = pred_df["eval_status"].value_counts()["FN"]
    count_all = len(list(pred_df["gold_label"]))

    FPs.append(FP)
    perc_FP.append(FP/count_all)

    FNs.append(FN)
    perc_FN.append(FN/count_all)
            
    print("0-Shot ", ty)
    #print(FPs)
    print(f"FP: {round(100*mean(perc_FP),2)}\\pm{round(100*np.std(np.asarray(perc_FP)),2)} ({mean(FPs)}\\pm {round(np.std(np.asarray(FPs)),2)})")
    #print(f"FP: {median(FPs)}\\pm\{round(np.std(np.asarray(FPs)),2)} ({100*median(perc_FP)}\\pm{100*np.std(np.asarray(perc_FP))})")

    #print(FNs)
    print(f"FN: {round(100*mean(perc_FN),2)}\\pm{round(100*np.std(np.asarray(perc_FN)),2)} ({mean(FNs)}\\pm\{round(np.std(np.asarray(FNs)),2)})")
    for num in nums:
        FPs = []
        FNs = []
        perc_FP = []
        perc_FN = []
        for seed in seeds:
            for run in range(1,4):
                #TODO adapt this path
                pred_df = pd.read_csv(f"/home/oenni/Dokumente/Self-ContradictionProject/Additional_Material_Coling/baselines/Llama3.3-70B/few-shot/{ty}/Llama3.3-70B_{ty}_{num}_{seed}_{run}.tsv", sep="\t").dropna()

                preds = [pred.lower().replace("**","").replace(".","").replace("_","").replace("'","").strip() for pred in list(pred_df["prediction"])]

                #print(set(preds))
                
                for i in range(len(preds)):
                    pred = preds[i]
                    if "self-contr" in pred:
                        preds[i] = "self-contradiction"
                    elif "non-contr" in pred:
                        preds[i] = "non-contradiction"

                #print(set(preds))

                loc_status = []

                pred_df["prediction"] = preds

                for gold, pred in zip(pred_df["gold_label"], preds):
                    #print(pred, type(pred))
                    if gold == "self-contradiction" and pred == "self-contradiction":
                        loc_status.append("TP")
                    elif gold == "non-contradiction" and pred == "non-contradiction":
                        loc_status.append("TN")
                    elif gold == "self-contradiction" and pred == "non-contradiction":
                        loc_status.append("FN")
                    elif gold == "non-contradiction" and pred == "self-contradiction":
                        loc_status.append("FP")

                pred_df["eval_status"] = loc_status

                preds_FP = pred_df[pred_df["eval_status"] == "FP"]
                preds_FN = pred_df[pred_df["eval_status"] == "FN"]

                preds_FN.to_csv(f"/home/oenni/Dokumente/Self-ContradictionProject/Additional_Material_Coling/baselines/Llama3.3-70B/erroranalysis/Llama3.3-70B_FN_{ty}_{num}_{seed}_{run}.tsv", sep="\t", index=False)
                preds_FP.to_csv(f"/home/oenni/Dokumente/Self-ContradictionProject/Additional_Material_Coling/baselines/Llama3.3-70B/erroranalysis/Llama3.3-70B_FP_{ty}_{num}_{seed}_{run}.tsv", sep="\t", index=False)

                FP = pred_df["eval_status"].value_counts()["FP"]
                FN = pred_df["eval_status"].value_counts()["FN"]
                count_all = len(list(pred_df["gold_label"]))

                FPs.append(FP)
                perc_FP.append(FP/count_all)

                FNs.append(FN)
                perc_FN.append(FN/count_all)
                
        print(num, "-Shot ", ty)
        #print(FPs)
        print(f"FP: {round(100*mean(perc_FP),2)}\\pm{round(100*np.std(np.asarray(perc_FP)),2)} ({mean(FPs)}\\pm {round(np.std(np.asarray(FPs)),2)})")
        #print(f"FP: {median(FPs)}\\pm\{round(np.std(np.asarray(FPs)),2)} ({100*median(perc_FP)}\\pm{100*np.std(np.asarray(perc_FP))})")

        #print(FNs)
        print(f"FN: {round(100*mean(perc_FN),2)}\\pm{round(100*np.std(np.asarray(perc_FN)),2)} ({mean(FNs)}\\pm\{round(np.std(np.asarray(FNs)),2)})")
        #print(f"FN: {median(FNs)}\\pm\{round(np.std(np.asarray(FNs)),2)} ({100*median(perc_FN)}\\pm{100*np.std(np.asarray(perc_FN))})")


types = ["locution", "proposition"]

#for ty in types:
#    FN_best = pd.read_csv(f"/home/oenni/Dokumente/Self-ContradictionProject/Additional_Material_Coling/baselines/Llama3.3-70B/erroranalysis/Llama3.3-70B_FN_{ty}_32_42_1.tsv", sep="\t")
#    FP_best = pd.read_csv(f"/home/oenni/Dokumente/Self-ContradictionProject/Additional_Material_Coling/baselines/Llama3.3-70B/erroranalysis/Llama3.3-70B_FP_{ty}_32_42_1.tsv", sep="\t")

#    FN_best_100 = FN_best.sample(random_state=42, frac=0.5).to_csv(f"/home/oenni/Dokumente/Self-ContradictionProject/Additional_Material_Coling/baselines/Llama3.3-70B/erroranalysis/Llama3.3-70B_FN_{ty}_32_42_1_100.tsv", index=False, sep="\t")
#    FP_best_100 = FP_best.sample(random_state=42, frac=0.5).to_csv(f"/home/oenni/Dokumente/Self-ContradictionProject/Additional_Material_Coling/baselines/Llama3.3-70B/erroranalysis/Llama3.3-70B_FP_{ty}_32_42_1_100.tsv", index=False, sep="\t")

