import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import random
import time
import jsonlines
from transformers import LEDTokenizer
import gensim.downloader as api
from tqdm import tqdm
import multiprocessing
import functools
import spacy
import sys

sys.path.append("../../")
from utils.metrics import rouge
from hgsum.model.dataloading import concatenate_documents, tokenize_tgs


def prepare_graph(concatenated_text, glove_wv, online=True, for_summary=False):
    docsep_token = "<doc-sep>"
    sentsep_token = "<sent-sep>"
    bos_token = "<s>"
    eos_token = "</s>"

    # positions to distinguish word, sent and doc token in input_ids
    tokens_positions = []
    sents_positions = []
    docs_positions = []

    tokens = []  # token nodes
    token_embeddings = []
    sents = []  # sentence nodes, every sent is a token list
    docs = []  # doc nodes, every doc is a sent list
    sent_tmp = []
    doc_tmp = []

    # sent-token, sent_contain_token
    sent_token_contain_l = []
    sent_token_contain_r = []
    sent_token_contain_w = []
    # doc-sent, doc_contain_sent
    doc_sent_contain_l = []
    doc_sent_contain_r = []
    doc_sent_contain_w = []

    for index, token in enumerate(concatenated_text):
        token = token.replace("Ġ", "")
        if token != docsep_token and token != sentsep_token and token != bos_token and token != eos_token:
            token_embeddings.append(glove_wv.get(token, [0]))
            tokens_positions.append(index)
            tokens.append(token)
            sent_tmp.append(token)

        if token == sentsep_token:
            sents_positions.append(index)
            sents.append(sent_tmp)

            tokens_len = len(tokens)
            sent_tmp_len = len(sent_tmp)
            sent_index = len(sents) - 1
            for i in range(tokens_len - sent_tmp_len, tokens_len):
                sent_token_contain_l.append(sent_index)
                sent_token_contain_r.append(i)
                sent_token_contain_w.append(1)

            doc_tmp.append(sent_tmp)
            sent_tmp = []

        if token == docsep_token:
            docs_positions.append(index)
            docs.append(doc_tmp)

            sents_len = len(sents)
            doc_tmp_len = len(doc_tmp)
            doc_index = len(docs) - 1
            for i in range(sents_len - doc_tmp_len, sents_len):
                doc_sent_contain_l.append(doc_index)
                doc_sent_contain_r.append(i)
                doc_sent_contain_w.append(1)

            doc_tmp = []
    # print("token-sent-doc")

    # token-token, similar_token (Glove)
    # https://radimrehurek.com/gensim/models/keyedvectors.html?highlight=glove
    token_token_similarity_l = []
    token_token_similarity_r = []
    token_token_similarity_w = []
    # token-token, followed_by (order information)
    token_token_follow_l = []
    token_token_follow_r = []
    token_token_follow_w = []

    for i, token_i_embedding in enumerate(token_embeddings):
        # token-token, similarity
        # if token_i_embedding==[0]:
        #     for j, token_j_embedding in enumerate(token_embeddings):
        #         if j > i and token_j_embedding == token_i_embedding:
        #             token_token_similarity_l.append(i)
        #             token_token_similarity_r.append(j)
        #             token_token_similarity_w.append(1)
        # else:
        if token_i_embedding != [0]:
            for j, token_j_embedding in enumerate(token_embeddings):
                if j > i:
                    if token_i_embedding == token_j_embedding:
                        token_token_similarity_l.append(i)
                        token_token_similarity_r.append(j)
                        token_token_similarity_w.append(1)
                    else:
                        if token_j_embedding != [0]:
                            similarity = cosine_similarity([token_i_embedding], [token_j_embedding])[0][0]
                            # print(type(similarity))
                            if similarity > 0.5:
                                token_token_similarity_l.append(i)
                                token_token_similarity_r.append(j)
                                token_token_similarity_w.append(float(similarity))  # float32 is not serialized

        # token-token, follow
        if i < len(tokens) - 1:
            token_token_follow_l.append(i)
            token_token_follow_r.append(i + 1)
            token_token_follow_w.append(1)
    # print("token-token")

    # sent-sent, similar_sent (rouge)
    sent_sent_rouge_l = []
    sent_sent_rouge_r = []
    sent_sent_rouge_w = []
    sents_stripped = [" ".join(sent) for sent in sents]

    for i, sent_i_tripped in enumerate(sents_stripped):
        for j, sent_j_tripped in enumerate(sents_stripped):
            if j > i:
                f1 = rouge(sent_i_tripped, sent_j_tripped, types=["rouge2"])["rouge2"]["fmeasure"]
                sent_sent_rouge_l.append(i)
                sent_sent_rouge_r.append(j)
                sent_sent_rouge_w.append(f1)

    # print("sent-sent")

    # doc-doc, similar_doc (Rouge)
    doc_doc_rouge_l = []
    doc_doc_rouge_r = []
    doc_doc_rouge_w = []
    docs_stripped = ["\n".join([" ".join(sent) for sent in doc]) for doc in docs]

    for i, doc_i_stripped in enumerate(docs_stripped):
        for j, doc_j_stripped in enumerate(docs_stripped):
            if j > i:
                f1 = rouge(doc_j_stripped, doc_i_stripped, types=["rougeLsum"])["rougeLsum"]["fmeasure"]
                doc_doc_rouge_l.append(i)
                doc_doc_rouge_r.append(j)
                doc_doc_rouge_w.append(f1)
    # print("doc-doc")

    heterograph_data = {}
    heterograph_data["token_token_similarity_l"] = token_token_similarity_l
    heterograph_data["token_token_similarity_r"] = token_token_similarity_r
    heterograph_data["token_token_similarity_w"] = token_token_similarity_w
    heterograph_data["token_token_follow_l"] = token_token_follow_l
    heterograph_data["token_token_follow_r"] = token_token_follow_r
    heterograph_data["token_token_follow_w"] = token_token_follow_w
    heterograph_data["sent_sent_rouge_l"] = sent_sent_rouge_l
    heterograph_data["sent_sent_rouge_r"] = sent_sent_rouge_r
    heterograph_data["sent_sent_rouge_w"] = sent_sent_rouge_w
    heterograph_data["sent_token_contain_l"] = sent_token_contain_l
    heterograph_data["sent_token_contain_r"] = sent_token_contain_r
    heterograph_data["sent_token_contain_w"] = sent_token_contain_w
    heterograph_data["tokens_positions"] = tokens_positions
    heterograph_data["sents_positions"] = sents_positions
    if not for_summary:
        heterograph_data["doc_sent_contain_l"] = doc_sent_contain_l
        heterograph_data["doc_sent_contain_r"] = doc_sent_contain_r
        heterograph_data["doc_sent_contain_w"] = doc_sent_contain_w
        heterograph_data["doc_doc_rouge_l"] = doc_doc_rouge_l
        heterograph_data["doc_doc_rouge_r"] = doc_doc_rouge_r
        heterograph_data["doc_doc_rouge_w"] = doc_doc_rouge_w
        heterograph_data["docs_positions"] = docs_positions
    if not online:
        heterograph_data["sents_tripped"] = sents_stripped
    return heterograph_data


def prepare_graph_multi_process_offline(i, samples, tokenizer, glove_wv):
    sample = samples[i]
    all_docs = sample["source_documents"]
    concatenated_text = concatenate_documents(all_docs, with_sent_sep=True, tokenizer=tokenizer, max_input_len=4096)
    heterograph_source = prepare_graph(concatenated_text, glove_wv, online=False, for_summary=False)

    tgt = sample["summary"]
    tokenized_tgt = tokenize_tgs(tgt, with_sent_sep=True, tokenizer=tokenizer, max_output_len=-1)
    heterograph_tgt = prepare_graph(tokenized_tgt, glove_wv, online=False, for_summary=True)

    sample["heterograph_source"] = heterograph_source
    sample["heterograph_tgt"] = heterograph_tgt
    return sample


if __name__ == "__main__":
    nlp = spacy.load('en_core_web_sm')

    dataset_name = "multinews"
    token_type = "noun"
    pretrained_primer = "allenai/PRIMERA"
    tokenizer = LEDTokenizer.from_pretrained(pretrained_primer)
    docsep_token_id = tokenizer.convert_tokens_to_ids("<doc-sep>")
    tokenizer.add_tokens(["<sent-sep>"])
    sentsep_token_id = tokenizer.convert_tokens_to_ids("<sent-sep>")

    save_file = "../data/%s_graph_%s.json" % (dataset_name, token_type)
    if os.path.exists(save_file):
        os.remove(save_file)

    samples = []
    with jsonlines.open("../../datasets/%s.json" % dataset_name) as reader:
        for sample in reader:
            samples.append(sample)
    random.seed(42)
    # samples = random.choices(samples, k=16)  # this is for debugging
    print("dataset loaded", len(samples))
    print("LED tokenizer vocabulary", len(tokenizer.get_vocab()))

    glove = api.load("glove-wiki-gigaword-100")
    glove_wv = {}
    for token in tokenizer.get_vocab():
        token_stripped = token.replace("Ġ", "")
        if token_stripped != "":
            doc = nlp(token_stripped)
            token = doc[0]
            if token.pos_ in ['NOUN'] and glove.has_index_for(token_stripped):
                glove_wv[token_stripped] = list(glove[token_stripped])
    print("GloVe wv loaded", len(glove_wv))

    # print(glove_wv.keys())
    start_time = time.time()

    processes = 4
    chunksize = 2
    len_samples_all = len(samples)
    step = processes * chunksize
    for start in range(0, len_samples_all, step):
        if start + step <= len_samples_all:
            step_samples = samples[start:start + step]
        else:
            step_samples = samples[start:len_samples_all]
        len_data = len(step_samples)
        partial_computing = functools.partial(prepare_graph_multi_process_offline, samples=step_samples,
                                              tokenizer=tokenizer, glove_wv=glove_wv)
        # with multiprocessing.Pool(os.cpu_count()) as p:
        with multiprocessing.Pool(processes=processes) as p:
            step_samples = list(
                tqdm(p.imap(partial_computing, range(len_data), chunksize=chunksize), total=len_data,
                     desc="preparing graph data"))

        # for sample in tqdm(step_samples):
        #     all_docs = sample["source_documents"]
        #     concatenated_text = concatenate_documents(all_docs, with_sent_sep=True, tokenizer=tokenizer, max_input_len=4096)
        #     words_positions, sents_positions, docs_positions, heterograph_data = prepare_graph(concatenated_text, glove_wv,
        #                                                                                         online=False)
        #     sample["words_positions"] = words_positions
        #     sample["sents_positions"] = sents_positions
        #     sample["docs_positions"] = docs_positions
        #     sample["heterograph_data"] = heterograph_data

        print("graph constructed", start)

        with jsonlines.open(save_file, "a") as writer:
            writer.write_all(step_samples)
    end_time = time.time()
    print((end_time - start_time) / 3600, "h")
