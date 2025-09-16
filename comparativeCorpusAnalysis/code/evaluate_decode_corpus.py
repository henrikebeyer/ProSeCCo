import pandas as pd
from transformers import pipeline

analysed_file = "/home/oenni/Dokumente/Self-ContradictionProject/decode_parsed/contradiction_df_random_10_perc.tsv"

df = pd.read_csv(analysed_file)

#for val in df.value_counts("contradiction type"):
#    print(val, round(val/sum(df.value_counts("contradiction type"))*100,2))

#print("\n")
#for val in df.value_counts("contradiction type"):
#    print(val, round(val/(sum(df.value_counts("contradiction type"))-258)*100,2))

df_antonymy = df[df["contradiction type"] == "antonymy"]

seg1_antonymy = df_antonymy["seg1"]
seg2_antonymy = df_antonymy["seg2"]

seg1_hate = []
seg2_hate = []

for seg1 in seg1_antonymy:
    if "hate" in seg1:
        seg1_hate.append(seg1)

for seg2 in seg2_antonymy:
    if "hate" in seg2:
        seg2_hate.append(seg2)



#print(len(seg1_hate), len(seg2_hate), round(100*(len(seg1_hate)+len(seg2_hate))/len(seg1_antonymy),2)) 

#print(round(df.value_counts("issues?")/len(df["seg1"])*100,2))

# run a sentiment analysis on the subset of decode
seg1 = list(df["seg1"])
seg2 = list(df["seg2"])

sentiment_analysis = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")

seg1_senti = []
seg2_senti = []

for seg1, seg2 in zip(seg1, seg2):
    seg1_senti.append(sentiment_analysis(seg1)[0]["label"])
    seg2_senti.append(sentiment_analysis(seg2)[0]["label"])

df["seg1 sentiment"] = seg1_senti
df["seg2 sentiment"] = seg2_senti

print(df.value_counts("seg1 sentiment"))
print(df.value_counts("seg2 sentiment"))

for num in df.value_counts("seg1 sentiment"):
    print(num, round(100*num/sum(df.value_counts("seg1 sentiment")),2))

for num in df.value_counts("seg2 sentiment"):
    print(num, round(100*num/sum(df.value_counts("seg2 sentiment")),2))

df.to_csv("/home/oenni/Dokumente/Self-ContradictionProject/decode_parsed/contradiction_df_random_10_perc_sentiment.tsv", index=False, sep="\t")

df["POS-NEG"] = [True if seg1=="POS" and seg2=="NEG" else False for seg1, seg2 in zip(df["seg1 sentiment"], df["seg2 sentiment"])]
print(df.value_counts("POS-NEG"))

df["NEG-POS"] = [True if seg1=="NEG" and seg2=="POS" else False for seg1, seg2 in zip(df["seg1 sentiment"], df["seg2 sentiment"])]
print(df.value_counts("NEG-POS"))

df.to_csv("/home/oenni/Dokumente/Self-ContradictionProject/decode_parsed/contradiction_df_random_10_perc_sentiment_check.tsv", index=False, sep="\t")
