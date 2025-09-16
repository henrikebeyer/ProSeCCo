import pandas as pd
from transformers import pipeline

path = "/home/oenni/Dokumente/Self-ContradictionProject/reannotation_CAPTURE_contradiction_types/CAPTURE_contradiction_types.tsv"

df = pd.read_csv(path)

#print(df.value_counts("contradiction type"))

#for val in df.value_counts("contradiction type"):
#    print(val, round(100*val/sum(df.value_counts("contradiction type")),2))

seg1 = df["proposition1"]
seg2 = df["proposition2"]

sentiment_analysis = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")

seg1_senti = []
seg2_senti = []

for seg1, seg2 in zip(seg1, seg2):
    seg1_senti.append(sentiment_analysis(seg1)[0]["label"])
    seg2_senti.append(sentiment_analysis(seg2)[0]["label"])

df["prop1 sentiment"] = seg1_senti
df["prop2 sentiment"] = seg2_senti

print(df.value_counts("prop1 sentiment"))
print(df.value_counts("prop2 sentiment"))

for num in df.value_counts("prop1 sentiment"):
    print(num, round(100*(num/sum(df.value_counts("prop1 sentiment"))),2))

for num in df.value_counts("prop2 sentiment"):
    print(num, round(100*num/sum(df.value_counts("prop2 sentiment")),2))

df["POS-NEG"] = [True if seg1=="POS" and seg2=="NEG" else False for seg1, seg2 in zip(df["prop1 sentiment"], df["prop2 sentiment"])]
print(df.value_counts("POS-NEG"))

df["NEG-POS"] = [True if seg1=="NEG" and seg2=="POS" else False for seg1, seg2 in zip(df["prop1 sentiment"], df["prop2 sentiment"])]
print(df.value_counts("NEG-POS"))

df.to_csv("/home/oenni/Dokumente/Self-ContradictionProject/reannotation_CAPTURE_contradiction_types/CAPTURE_contradiction_types_sentiment.tsv", index=False, sep="\t")

