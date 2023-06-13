import os
import random

import numpy as np
import sys
sys.path.append("../../")
from utils.metrics import rouge
import re
from nltk.tokenize import sent_tokenize

def read_folder(dir):
    files = os.listdir(dir)
    results = []
    for file in files:
        if dir not in file:
            file = os.path.join(dir, file)
        with open(file) as f:
            print(file)
            all = f.readlines()
            if "#*#*#*#*#*prediction#*#*#*#*#\n" in all:
                prediction_index = all.index("#*#*#*#*#*prediction#*#*#*#*#\n")
                reference_index = all.index("#*#*#*#*#*reference#*#*#*#*#\n")
                source_index = all.index("#*#*#*#*#*source documents#*#*#*#*#\n")
                prediction = " ".join(" ".join(all[prediction_index+1: reference_index]).split()[:500])
                prediction = re.sub('[^A-Za-z,0-9.\'!?;():\s]+', '', prediction)
                sentences = sent_tokenize(prediction)
                random.shuffle(sentences)
                prediction = " ".join(sentences)
                reference = " ".join(" ".join(all[reference_index+1: source_index]).split())
                reference = re.sub('[^A-Za-z,0-9.!?;\'():\s]+', '', reference)
                sentences = sent_tokenize(reference)
                random.shuffle(sentences)
                reference = " ".join(sentences)
                results.append([prediction, reference])
            else:
                prediction = " ".join(" ".join(all).split()[:500])
                prediction = re.sub('[^A-Za-z,0-9.!?;\'():\s]+', '', prediction)
                sentences = sent_tokenize(prediction)
                random.shuffle(sentences)
                prediction = " ".join(sentences)
                results.append([prediction, ""])

    return results[:64]

dir = "/home/miao4/punim0521/NeuralAbstractiveSummarization/opensource/primer/PRIMER-oaimli/results/hgsum_multinews/generated_txt_0_multi_news_4096_1024_beam=5"
hgsum_multinews_results = read_folder(dir)
dir = "/home/miao4/punim0521/NeuralAbstractiveSummarization/opensource/primer/PRIMER-oaimli/results/primera_multinews/generated_txt_0_multi_news_beam=5_4096_1024"
primera_multinews_results = read_folder(dir)
dir = "/home/miao4/punim0521/NeuralAbstractiveSummarization/opensource/primer/PRIMER-oaimli/results/led_multinews/generated_txt_0_multi_news_4096_1024_beam=2"
led_multinews_results = read_folder(dir)
dir = "/home/miao4/punim0521/NeuralAbstractiveSummarization/opensource/primer/PRIMER-oaimli/results/pegasus_multinews/generated_txt_0_multi_news_1024_512_beam=5"
pegasus_multinews_results = read_folder(dir)
dir = "/home/miao4/punim0521/NeuralAbstractiveSummarization/opensource/primer/PRIMER-oaimli/results/graphsum_multinews/generated_txt_0_multi_news_beam=1_4050_300"
graphsum_multinews_results = read_folder(dir)
dir = "/home/miao4/punim0521/NeuralAbstractiveSummarization/opensource/primer/PRIMER-oaimli/results/mgsum_multinews/generated_txt_0_multi_news_512_400_beam=1"
mgsum_multinews_results = read_folder(dir)

multinews_final = []
id = 0
for hgsum, primera, led, pegasus, graphsum, mgsum in zip(hgsum_multinews_results, primera_multinews_results, led_multinews_results, pegasus_multinews_results, graphsum_multinews_results, mgsum_multinews_results):
    id += 1
    multinews_final.append("\multicolumn{2}{c}{\\textbf{Sample %d}}\\\\ \n \\midrule\n"%id)
    multinews_final.append("Ground-truth & %s \\\\ \n" % hgsum[1])
    multinews_final.append("\midrule \n")
    multinews_final.append("\\hgsum & %s \\\\ \n" % hgsum[0])
    multinews_final.append("\midrule \n")
    # multinews_final.append("\\hgsum & %s \\\\ \n" % hgsum[1])
    multinews_final.append("PRIMERA & %s \\\\ \n" % primera[0])
    multinews_final.append("\midrule \n")
    # multinews_final.append("PRIMERA & %s \\\\ \n" % primera[1])
    multinews_final.append("LED & %s \\\\ \n" % led[0])
    multinews_final.append("\midrule \n")
    # multinews_final.append("LED & %s \\\\ \n" % led[1])
    multinews_final.append("PEGASUS & %s \\\\ \n" % pegasus[0])
    multinews_final.append("\midrule \n")
    # multinews_final.append("PEGASUS & %s \\\\ \n" % pegasus[1])
    multinews_final.append("GraphSum & %s \\\\ \n" % graphsum[0])
    multinews_final.append("\midrule \n")
    # multinews_final.append("GraphSum & %s \\\\ \n" % graphsum[1])
    multinews_final.append("MGSum & %s \\\\ \n" % mgsum[0])
    multinews_final.append("\midrule \n")
    # multinews_final.append("MGSum & %s \\\\ \n" % mgsum[1])
    multinews_final.append("\midrule \n")

with open("generated_summaries_multinews.txt", 'w') as f:
    f.writelines(multinews_final)



dir = "/home/miao4/punim0521/NeuralAbstractiveSummarization/opensource/primer/PRIMER-oaimli/results/hgsum_wcep/generated_txt_0_wcep_4096_1024_beam=5"
hgsum_wcep_results = read_folder(dir)
dir = "/home/miao4/punim0521/NeuralAbstractiveSummarization/opensource/primer/PRIMER-oaimli/results/primera_wcep/generated_txt_0_wcep_beam=5_4096_1024"
primera_wcep_results = read_folder(dir)
dir = "/home/miao4/punim0521/NeuralAbstractiveSummarization/opensource/primer/PRIMER-oaimli/results/led_wcep/generated_txt_0_wcep_4096_1024_beam=2"
led_wcep_results = read_folder(dir)
dir = "/home/miao4/punim0521/NeuralAbstractiveSummarization/opensource/primer/PRIMER-oaimli/results/pegasus_wcep/generated_txt_0_wcep_1024_512_beam=5"
pegasus_wcep_results = read_folder(dir)
dir = "/home/miao4/punim0521/NeuralAbstractiveSummarization/opensource/primer/PRIMER-oaimli/results/graphsum_wcep/generated_txt_0_wcep_beam=1_4050_300"
graphsum_wcep_results = read_folder(dir)
dir = "/home/miao4/punim0521/NeuralAbstractiveSummarization/opensource/primer/PRIMER-oaimli/results/mgsum_wcep/generated_txt_0_wcep_512_400_beam=1"
mgsum_wcep_results = read_folder(dir)

multinews_final = []
id = 0
for hgsum, primera, led, pegasus, graphsum, mgsum in zip(hgsum_wcep_results, primera_wcep_results, led_wcep_results, pegasus_wcep_results, graphsum_wcep_results, mgsum_wcep_results):
    id += 1
    multinews_final.append("\multicolumn{2}{c}{\\textbf{Sample %d}}\\\\ \n \\midrule\n" % id)
    multinews_final.append("Ground-truth & %s \\\\ \n" % hgsum[1])
    multinews_final.append("\midrule \n")
    multinews_final.append("\\hgsum & %s \\\\ \n" % hgsum[0])
    multinews_final.append("\midrule \n")
    # multinews_final.append("\\hgsum & %s \\\\ \n" % hgsum[1])
    multinews_final.append("PRIMERA & %s \\\\ \n" % primera[0])
    multinews_final.append("\midrule \n")
    # multinews_final.append("PRIMERA & %s \\\\ \n" % primera[1])
    multinews_final.append("LED & %s \\\\ \n" % led[0])
    multinews_final.append("\midrule \n")
    # multinews_final.append("LED & %s \\\\ \n" % led[1])
    multinews_final.append("PEGASUS & %s \\\\ \n" % pegasus[0])
    multinews_final.append("\midrule \n")
    # multinews_final.append("PEGASUS & %s \\\\ \n" % pegasus[1])
    multinews_final.append("GraphSum & %s \\\\ \n" % graphsum[0])
    multinews_final.append("\midrule \n")
    # multinews_final.append("GraphSum & %s \\\\ \n" % graphsum[1])
    multinews_final.append("MGSum & %s \\\\ \n" % mgsum[0])
    multinews_final.append("\midrule \n")
    # multinews_final.append("MGSum & %s \\\\ \n" % mgsum[1])
    multinews_final.append("\midrule \n")

with open("generated_summaries_wcep.txt", 'w') as f:
    f.writelines(multinews_final)



dir = "/home/miao4/punim0521/NeuralAbstractiveSummarization/opensource/primer/PRIMER-oaimli/results/hgsum_arxiv/generated_txt_0_arxiv_beam=1_16384_1024"
hgsum_arxiv_results = read_folder(dir)
dir = "/home/miao4/punim0521/NeuralAbstractiveSummarization/opensource/primer/PRIMER-oaimli/results/primera_arxiv/generated_txt_0_arxiv_4096_1024_beam=1"
primera_arxiv_results = read_folder(dir)
dir = "/home/miao4/punim0521/NeuralAbstractiveSummarization/opensource/primer/PRIMER-oaimli/results/led_arxiv/generated_txt_0_arxiv_beam=5_4096_1024"
led_arxiv_results = read_folder(dir)
dir = "/home/miao4/punim0521/NeuralAbstractiveSummarization/opensource/primer/PRIMER-oaimli/results/pegasus_arxiv/generated_txt_0_arxiv_1024_512_beam=1"
pegasus_arxiv_results = read_folder(dir)
dir = "/home/miao4/punim0521/NeuralAbstractiveSummarization/opensource/primer/PRIMER-oaimli/results/graphsum_arxiv/generated_txt_0_arxiv_beam=1_3000_300"
graphsum_arxiv_results = read_folder(dir)
dir = "/home/miao4/punim0521/NeuralAbstractiveSummarization/opensource/primer/PRIMER-oaimli/results/mgsum_arxiv/generated_txt_0_arxiv_beam=1_4050_400"
mgsum_arxiv_results = read_folder(dir)

multinews_final = []
id = 0
for hgsum, primera, led, pegasus, graphsum, mgsum in zip(hgsum_arxiv_results, primera_arxiv_results, led_arxiv_results, pegasus_arxiv_results, graphsum_arxiv_results, mgsum_arxiv_results):
    id += 1
    multinews_final.append("\multicolumn{2}{c}{\\textbf{Sample %d}}\\\\ \n \\midrule\n" % id)
    multinews_final.append("Ground-truth & %s \\\\ \n" % pegasus[1])
    multinews_final.append("\midrule \n")
    multinews_final.append("\\hgsum & %s \\\\ \n" % hgsum[0])
    multinews_final.append("\midrule \n")
    # multinews_final.append("\\hgsum & %s \\\\ \n" % hgsum[1])
    multinews_final.append("PRIMERA & %s \\\\ \n" % primera[0])
    multinews_final.append("\midrule \n")
    # multinews_final.append("PRIMERA & %s \\\\ \n" % primera[1])
    multinews_final.append("LED & %s \\\\ \n" % led[0])
    multinews_final.append("\midrule \n")
    # multinews_final.append("LED & %s \\\\ \n" % led[1])
    multinews_final.append("PEGASUS & %s \\\\ \n" % pegasus[0])
    multinews_final.append("\midrule \n")
    # multinews_final.append("PEGASUS & %s \\\\ \n" % pegasus[1])
    multinews_final.append("GraphSum & %s \\\\ \n" % graphsum[0])
    multinews_final.append("\midrule \n")
    # multinews_final.append("GraphSum & %s \\\\ \n" % graphsum[1])
    multinews_final.append("MGSum & %s \\\\ \n" % mgsum[0])
    multinews_final.append("\midrule \n")
    # multinews_final.append("MGSum & %s \\\\ \n" % mgsum[1])
    multinews_final.append("\midrule \n")

with open("generated_summaries_arxiv.txt", 'w') as f:
    f.writelines(multinews_final)

