import pandas as pd

corpus = pd.read_csv("SelfContra_annot/SelfContra_ordered_annot.tsv", sep="\t")
#annotC = pd.read_csv("SelfContra_annot/SelfContra_myannot_full_ordered.tsv", sep="\t")



annotations1 = list(corpus["label"])
annotations2 = list(corpus["mylabel"])

print(set(annotations1))
#annotations2 = [label.lower() for label in annotB["label"]]

TP = 0
TN = 0
FP = 0
FN = 0

# calculate TP, TN, FP, FN
# TP = times both annotators agreed on "contradiction" label
# TN = times both annotators agreed on  "non-contradiction" label
# FP = number of times annotator1 annotated "contradiction" label and annotator2 annotated "non-contradiction" label
# FN = number of times annotator1 annotated "non-contradiction" label and annotator2 annotated "contradiction" label
for annot1, annot2 in zip(annotations1, annotations2):
    if annot1 == 1 and annot2 == 1:
        TP += 1
    elif annot1 == 0 and annot2 == 0: 
        TN += 1
    elif annot1 == 1 and annot2 == 0:
        FP += 1
    elif annot1 == 0 and annot2 == 1:
        FN += 1

# n = total number of samples
n = len(annotations1)


P_0 = (TP + TN) / n
P1 = ((TP + FN) * (TP + FP)) / n**2
P2 = ((TN + FN) * (TN +FP)) / n**2
P_e = P1 + P2

chohens_K = (P_0 - P_e) / (1 - P_e)

print(chohens_K)