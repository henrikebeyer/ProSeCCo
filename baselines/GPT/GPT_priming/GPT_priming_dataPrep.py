import pandas as pd
import random
import csv

seed = 52 #42, 121, 78, 52
random.seed(seed)

corpus_file = "/home/oenni/Dokumente/Self-ContradictionProject/Additional_Material_Coling/ProSeCCo/fullCorpus/ProSeCCo_final.tsv"

data = pd.read_csv(corpus_file, sep="\t")

ty = "locution" #"locution"

statements1 = list(data[f"{ty}1"])
statements2 = list(data[f"{ty}2"])
labels = ["non-contradiction" if label == 0 else "self-contradiction" for label in list(data["label"])]
ids = list(data["ID"])

statement_pairs = []
 
for i in range(len(statements1)):
    statement1 = statements1[i]
    statement2 = statements2[i]
    statement_pair = f"'{statement1}', '{statement2}'"
    statement_pairs.append(statement_pair)


index_nums = [4] #[2, 8, 16, 32]
prime_ind_list = []

for i in range(len(index_nums)):
    index_num = index_nums[i]
    index_list = []
    while(len(index_list) != index_num):
        rand_int = random.randint(0, len(statement_pairs))
        if rand_int not in index_list:
            index_list.append(rand_int)
    prime_ind_list.append(index_list)

primes = []
samples = []
sample_labels = []
sample_ids = []

for i in range(len(index_nums)):
    prime_inds = prime_ind_list[i]
    prime_list = []
    sample_list = []
    label_list = []
    id_list = []
    for ind in prime_inds:
        #print(i, ind, len(statement_pairs))
        prime_list.append([ids[ind], statement_pairs[ind], labels[ind]])
        sample_list = [samp for idx, samp in enumerate(statement_pairs) if idx not in prime_inds]
        label_list = [samp for idx, samp in enumerate(labels) if idx not in prime_inds]
        id_list = [samp for idx, samp in enumerate(ids) if idx not in prime_inds]

        #print(len(sample_list))
    primes.append(prime_list)
    samples.append(sample_list)
    sample_labels.append(label_list)
    sample_ids.append(id_list)

for i in range(len(primes)):
    sample_num = index_nums[i]
    prime_list = primes[i]
    out_primes = f"/home/oenni/Dokumente/Self-ContradictionProject/Additional_Material_Coling/data/primes/{ty}_primes_{sample_num}_{seed}.tsv"
    with open(out_primes, "w") as fout:
        writer = csv.writer(fout, delimiter="\t")
        header = ["ID", "prime_sample", "label"]
        writer.writerow(header)
        for prime in prime_list:
            writer.writerow(prime)




for i in range(len(samples)):
    sample_num = index_nums[i]
    sample_list = samples[i]
    id_list = sample_ids[i]
    label_list = sample_labels[i]
    out_samples = f"/home/oenni/Dokumente/Self-ContradictionProject/Additional_Material_Coling/data/samples/{ty}_samples_{sample_num}_{seed}.tsv"
    with open(out_samples, "w") as fout:
        writer = csv.writer(fout, delimiter="\t")
        header = ["ID", "test_sample", "label"]
        writer.writerow(header)
        for j in range(len(sample_list)):
            writer.writerow([id_list[j], sample_list[j], label_list[j]])