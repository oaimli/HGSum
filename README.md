# Compressed Heterogeneous Graph for Abstractive Multi-document Summarization
[![arXiv](https://img.shields.io/badge/arxiv-2303.06565-brightgreen)](https://arxiv.org/pdf/2303.06565.pdf)

### Overview
Multi-document summarization (MDS) aims to generate a summary for a number of related documents. We propose HGSum — an MDS model that extends an encoder-decoder architecture to incorporate a heterogeneous graph to represent different semantic units (e.g., words and sentences) of the documents. This contrasts with existing MDS models which do not consider different edge types of graphs and as such do not capture the diversity of relationships in the documents. To preserve only key information and relationships of the documents in the heterogeneous graph, HGSum uses graph pooling to compress the input graph. And to guide HGSUM to learn the compression, we introduce an additional objective that maximizes the similarity between the compressed graph and the graph constructed from the ground-truth summary during training. HGSum is trained end-to-end with the graph similarity and standard cross-entropy objectives. 

### Architecure of HGSum
At its core, HGSUM extends a text encoder-decoder architecture to incorporate information from a compressed heterogeneous graph derived from the input source documents, as presented in Figure 2. HGSUM has four main components: (1) text encoder (initialized using PRIMERA weights), (2) graph encoder, (3) graph compressor, and (4) text decoder (initialized using PRIMERA weights). Please find more details about the architecture and training strategies in our paper.
![image](https://github.com/oaimli/HGSum/assets/12547070/8ce136c7-6adf-4480-af56-9ab8335d6239)


If you are going to use our dataset in your work, please cite our paper:

[Li et al. 2023] Miao Li, Jianzhong Qi, and Jey Han Lau. "Compressed Heterogeneous Graph for Abstractive Multi-Document Summarization". AAAI, 2023.
```
@inproceedings{peersum_2023,
  title={Compressed Heterogeneous Graph for Abstractive Multi-Document Summarization},
  author={Miao Li, Jianzhong Qi, and Jey Han Lau},
  booktitle={AAAI},
  year={2023}
}
```
