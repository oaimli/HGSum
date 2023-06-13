import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import pi

# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = "selif"

def plot_summarization_results():
    ks = np.arange(0, 1, step=0.1)

    rouge_avg_multinews = [0, 0, 0, 0, 0, 239.3, 239.3, 0, 0, 0,]
    rouge_avg_wcep_100 = [0, 0, 0, 0, 0, 0, 0, 161.8, 0, 0,]
    rouge_avg_arxiv = [0, 0, 0, 0, 26.3, 0, 0, 0, 0, 0,]

    plt.figure(figsize=(8,3.5))
    line1, = plt.plot(ks, rouge_avg_multinews, color='deepskyblue', linestyle='-', linewidth='1.5', label=r"\textsc{Multi-News}")
    line2, = plt.plot(ks, rouge_avg_wcep_100, color="darkcyan", linestyle=':', linewidth='1.5', label=r"\textsc{WCEP-100}")
    line3, = plt.plot(ks, rouge_avg_arxiv, color="brown", linestyle='--', linewidth='1.5', label=r"\textsc{arxiv}")

    plt.xlabel("{Compression ratio} $k$", fontsize=16, family='Times New Roman')
    plt.ylabel("Average length", fontsize=16, family='Times New Roman')
    plt.legend(handles=[line1, line2, line3], ncol=3, prop={"family": 'Times New Roman', "size": 13.5})

    plt.subplots_adjust(top=0.97, bottom=0.16, right=0.98, left=0.1)
    # plt.show()
    plt.savefig('summarization_results.png', dpi=1024)

def plot_bar_chart():
    ks = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    xs = range(1, 11)
    xs_psv1 = (np.array(xs) - 0.2).tolist()
    xs_psv2 = (np.array(xs) - 0.0).tolist()
    xs_psv3 = (np.array(xs) + 0.2).tolist()
    print(xs_psv1)

    rouge_avg_multinews = [38.0, 48.9, 67.6, 90.5, 164.7, 234.3, 243.0, 251.8, 256.1, 257.2]
    rouge_avg_wcep_100 = [18.0, 21.7, 21.8, 28.4, 26.3, 29.5, 28.4, 30.6, 31.8, 39.2]
    rouge_avg_arxiv = [100.6, 130.5, 149.4, 148.5, 160.5, 169.7, 182.1, 186.8, 189.7, 194.5]

    bar_width = 0.2
    plt.figure(figsize=(9, 4))
    plt.bar(xs_psv1, height=rouge_avg_wcep_100, width=bar_width, color='deepskyblue', label="WCEP-100")
    plt.bar(xs_psv3, height=rouge_avg_multinews, width=bar_width, color='darkcyan', label="Multi-News")
    plt.bar(xs_psv2, height=rouge_avg_arxiv, width=bar_width, color="orange", label="Arxiv")
    plt.xticks(xs, ks, fontproperties='Times New Roman', fontsize=24)
    plt.yticks(fontproperties='Times New Roman', fontsize=24)
    plt.xlabel("Compression ratio", fontdict={"size":24, "family": 'Times New Roman'})
    plt.ylabel("Average length", fontsize=24, family='Times New Roman')
    plt.legend(bbox_to_anchor=(0.94, -0.25), ncol=3, prop={"family": 'Times New Roman', "size": 20})
    plt.subplots_adjust(top=0.97, bottom=0.33, right=0.98, left=0.11)
    # plt.show()
    plt.savefig('summarization_results.png', dpi=1024)

if __name__=="__main__":
    plot_bar_chart()