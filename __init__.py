# Heterogeneous Graph Attentions for Abstractive Multi-document Summarization
# Leverage graph representation in PLMs for summarization

# Related codes from following repositories:
# Transformer, https://github.com/bentrevett/pytorch-seq2seq,
# PyGAT, https://github.com/Diego999/pyGAT
# HGAT, https://github.com/BUPT-GAMMA/HGAT
# Graph4NLP, https://github.com/graph4ai/graph4nlp
# DGL, https://github.com/dmlc/dgl
# Graph2Seq, https://github.com/IBM/Graph2Seq

# Training code here is based on the PRIMER code primer_hf_main.py from https://github.com/allenai/PRIMER,
# Implementation of the encoder-decoder model architecture is based on Transformers, https://github.com/huggingface/transformers/tree/main/src/transformers/models/led
# Graph construction and leveraging is based on https://github.com/jinpeng01/AIG_CL, https://github.com/zshicode/GNN-for-text-classification, https://github.com/iworldtong/text_gcn.pytorch

# Datasets:
# arxiv, pubmed, bigpatent
# multinews, wcep_10, wcep_100, multixscience
