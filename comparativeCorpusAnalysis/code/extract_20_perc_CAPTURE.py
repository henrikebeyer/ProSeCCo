import pandas as pd

path = "/home/oenni/Dokumente/Self-ContradictionProject/reannotation_CAPTURE_contradiction_types/CAPTURE_full_prop.tsv"

df = pd.read_csv(path, sep="\t")

contradictions = df[df["label"]==1.0]

contradictions.to_csv("/home/oenni/Dokumente/Self-ContradictionProject/reannotation_CAPTURE_contradiction_types/CAPTURE_contradiction_types.tsv", sep="\t")