from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import jsonlines
import time


def get_embeddings(samples, sbert, dataset_name, token_type):
    sents_stripped_all_source = []
    for sample in samples:
        sents_stripped_all_source.extend(sample["heterograph_source"]["sents_tripped"])
    print("sents_stripped_all_source", len(sents_stripped_all_source))
    # Start the multi-process pool on all available CUDA devices
    pool = sbert.start_multi_process_pool(target_devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    # Compute the embeddings using the multi-process pool
    sents_stripped_embeddings_source = sbert.encode_multi_process(sents_stripped_all_source, pool=pool, batch_size=64)
    sbert.stop_multi_process_pool(pool)
    print("sent embeddings of source documents done", len(sents_stripped_embeddings_source))

    sents_stripped_all_tgt = []
    for sample in samples:
        sents_stripped_all_tgt.extend(sample["heterograph_tgt"]["sents_tripped"])
    print("sents_stripped_all_tgt", len(sents_stripped_all_tgt))
    # Start the multi-process pool on all available CUDA devices
    pool = sbert.start_multi_process_pool(target_devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    # Compute the embeddings using the multi-process pool
    sents_stripped_embeddings_tgt = sbert.encode_multi_process(sents_stripped_all_tgt, pool=pool, batch_size=64)
    sbert.stop_multi_process_pool(pool)
    print("sent embeddings of summary done", len(sents_stripped_embeddings_tgt))

    sent_index_source = 0
    sent_index_tgt = 0
    samples_processed = []
    for sample in tqdm(samples, desc="computing sent-sent similarity"):
        sents_source = sample["heterograph_source"]["sents_tripped"]
        sents_embeddings = sents_stripped_embeddings_source[sent_index_source: sent_index_source + len(sents_source)]
        print("sents_embeddings source", len(sents_embeddings), len(sents_source))
        sent_sent_similarity_l = []
        sent_sent_similarity_r = []
        sent_sent_similarity_w = []
        for i, em_i in enumerate(sents_embeddings):
            for j, em_j in enumerate(sents_embeddings):
                if j > i:
                    similarity = cosine_similarity([em_i], [em_j])[0][0]
                    sent_sent_similarity_l.append(i)
                    sent_sent_similarity_r.append(j)
                    sent_sent_similarity_w.append(float(similarity))
        sample["heterograph_source"]["sent_sent_similarity_l"] = sent_sent_similarity_l
        sample["heterograph_source"]["sent_sent_similarity_r"] = sent_sent_similarity_r
        sample["heterograph_source"]["sent_sent_similarity_w"] = sent_sent_similarity_w
        sample["heterograph_source"]["sents_embeddings"] = sents_embeddings.tolist()
        sent_index_source += len(sents_source)

        sents_tgt = sample["heterograph_tgt"]["sents_tripped"]
        sents_embeddings = sents_stripped_embeddings_tgt[sent_index_tgt:sent_index_tgt + len(sents_tgt)]
        print("sents_embeddings tgt", len(sents_embeddings), len(sents_tgt))
        sent_sent_similarity_l = []
        sent_sent_similarity_r = []
        sent_sent_similarity_w = []
        for i, em_i in enumerate(sents_embeddings):
            for j, em_j in enumerate(sents_embeddings):
                if j > i:
                    similarity = cosine_similarity([em_i], [em_j])[0][0]
                    sent_sent_similarity_l.append(i)
                    sent_sent_similarity_r.append(j)
                    sent_sent_similarity_w.append(float(similarity))
        sample["heterograph_tgt"]["sent_sent_similarity_l"] = sent_sent_similarity_l
        sample["heterograph_tgt"]["sent_sent_similarity_r"] = sent_sent_similarity_r
        sample["heterograph_tgt"]["sent_sent_similarity_w"] = sent_sent_similarity_w
        sample["heterograph_tgt"]["sents_embeddings"] = sents_embeddings.tolist()
        sent_index_tgt += len(sents_tgt)

        samples_processed.append(sample)

    return samples_processed


if __name__ == "__main__":
    import os

    # # compute sent-sent similarity with sbert
    sbert = SentenceTransformer("all-mpnet-base-v2", device="cuda")
    print("SentenceTransformers loaded")

    dataset_name = "multinews"
    token_type = "noun"

    save_file = "../data/%s_graph_%s_sentem.json" % (dataset_name, token_type)
    if os.path.exists(save_file):
        os.remove(save_file)

    all_count = 0
    with jsonlines.open("../data/%s_graph_%s.json" % (dataset_name, token_type)) as reader:
        for sample in reader:
            all_count += 1

    start_time = time.time()
    samples = []
    index = 0
    with jsonlines.open("../data/%s_graph_%s.json" % (dataset_name, token_type)) as reader:
        for sample in reader:
            samples.append(sample)
            index += 1

            if len(samples) > 0 and len(samples) % 2000 == 0:
                print("data loaded", len(samples))
                # used to debug
                # samples = samples[:10]
                samples_processed = get_embeddings(samples, sbert, dataset_name, token_type)
                with jsonlines.open(save_file, "a") as writer:
                    writer.write_all(samples_processed)
                samples = []
                samples_processed = []

            if index == all_count:
                print("data loaded", len(samples))
                samples_processed = get_embeddings(samples, sbert, dataset_name, token_type)
                with jsonlines.open(save_file, "a") as writer:
                    writer.write_all(samples_processed)
                samples = []
                samples_processed = []
    # long running
    # do something other
    end_time = time.time()
    print((end_time - start_time) / 3600, "h")
