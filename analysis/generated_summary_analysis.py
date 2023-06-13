import os
import random
import sys
from evaluate import load
import numpy as np
import jsonlines

sys.path.append("../../")
from utils.metrics import rouge_corpus, rouge
bertscorer = load("bertscore")
# result_folder = "/home/miao4/punim0521/NeuralAbstractiveSummarization/opensource/primer/PRIMER/models/PRIMER_multinews/generated_txt_0_multi_news_4096_1024_beam=5"
result_folder = "/home/miao4/punim0521/NeuralAbstractiveSummarization/opensource/primer/PRIMER/models/PRIMER_arxiv/generated_txt_0_arxiv_4096_1024_beam=1"
# result_folder = "/home/miao4/punim0521/NeuralAbstractiveSummarization/opensource/primer/PRIMER/models/PRIMER_wcep/generated_txt_0_wcep_4096_1024_beam=5"
files = os.listdir(result_folder)
print("all samples", len(files))

references = []
predictions = []
document_clusters = []
for file in files:
    reference = ""
    prediction = ""
    source_documents = ""
    if not os.path.isdir(file):
        with open(os.path.join(result_folder, file)) as f:
            lines = f.readlines()
        prediction = " ".join(lines)
        # print(lines)
        # for i, l in enumerate(lines):
        #     l = l.strip()
        #     if l=="#*#*#*#*#*reference#*#*#*#*#":
        #         reference = lines[i+1]
        #     if l=="#*#*#*#*#*prediction#*#*#*#*#":
        #         prediction = lines[i+1]
        #     if l=="#*#*#*#*#*source documents#*#*#*#*#":
        #         source_documents = " ".join(lines[i+1:])
    # documents = source_documents[3:].split("<doc-sep>")[:-1]

    predictions.append(prediction)
    references.append("")
    document_clusters.append("")

# rouges = rouge_corpus(references=references, candidates=predictions, use_stemmer=True,
#           types=["rouge1", "rouge2", "rougeL", "rougeLsum"], split_summaries=True)
# bertscores = bertscorer.compute(predictions=predictions, references=references, lang="en")
#
# print(rouges)
# print("bertscore ", "precision", np.mean(bertscores["precision"]), "recall", np.mean(bertscores["recall"]), "f1", np.mean(bertscores["f1"]))

results = []
for x, y, z in zip(predictions, references, document_clusters):
    if len(x.split())<512:
        results.append({"prediction": x, "reference": y, "source_documents": z})

length_prediction = []
for result in results:
    length_prediction.append(len(result["prediction"].split()))
print("average length", np.mean(length_prediction))


# valid_results = []
# for result in results:
#     prediction = result["prediction"]
#     reference = result["reference"]
#     source_documents = result["source_documents"]
#
#     rouges = rouge(reference=reference, candidate=prediction, use_stemmer=True,
#               types=["rouge1", "rouge2", "rougeL", "rougeLsum"], split_summaries=True)
#     if rouges["rouge1"]["fmeasure"]>0.49 and rouges["rouge2"]["fmeasure"]>0.20 and rouges["rougeLsum"]["fmeasure"]>0.44 and len(source_documents)>=4:
#         print(rouges)
#         valid_results.append(result)
# print("valid results", len(valid_results))
# with jsonlines.open("generated_summaries_multinews_4.json", "w") as writer:
    # writer.write_all(valid_results)






