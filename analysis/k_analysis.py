import random
import numpy as np

import jsonlines
from nltk.tokenize import sent_tokenize
import sys

sys.path.append("../../")
from utils.metrics import rouge

dataset_name = "arxiv.json"
data_path = "../../datasets/"
samples = []
with jsonlines.open(data_path+dataset_name) as reader:
    for line in reader:
        samples.append(line)
samples = random.sample(samples, 512)

redundancy_proportion = []
related_proportion = []
for sample in samples:
    reference_summary = sample["summary"]
    source_documents = sample["source_documents"]
    source_documents_all = " ".join(source_documents)
    sentences = sent_tokenize(source_documents_all)

    unique_sentences = set(sentences)
    redundancy_proportion.append(len(unique_sentences)/len(sentences))

    relatedness_sentences = []
    for sentence in sentences:
        rouges = rouge(reference=reference_summary, candidate=sentence, split_summaries=True)
        relatedness = (rouges["rouge1"]["precision"] + rouges["rouge2"]["precision"] + rouges["rougeLsum"]["precision"])/3
        if relatedness>0.2:
            relatedness_sentences.append(relatedness)
    related_proportion.append(len(relatedness_sentences)/len(sentences))
    # related_proportion.append(np.mean(relatedness_sentences))

print(dataset_name)
print("related_proportion", np.mean(related_proportion))
print("redundancy_proportion", np.mean(redundancy_proportion))



