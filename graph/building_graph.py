import random
import torch
from torch_geometric.data import HeteroData


def build_graph(heterograph_data, for_summary=False):
    # print("sent_sent_similarity_l", len(heterograph_data["sent_sent_similarity_l"]))
    # print("sent_sent_similarity_r", len(heterograph_data["sent_sent_similarity_r"]))
    # print("sent_sent_similarity_w", len(heterograph_data["sent_sent_similarity_w"]))
    token_token_similarity_l = heterograph_data["token_token_similarity_l"]
    token_token_similarity_r = heterograph_data["token_token_similarity_r"]
    token_token_similarity_w = heterograph_data["token_token_similarity_w"]
    token_token_follow_l = heterograph_data["token_token_follow_l"]
    token_token_follow_r = heterograph_data["token_token_follow_r"]
    token_token_follow_w = heterograph_data["token_token_follow_w"]
    sent_sent_rouge_l = heterograph_data["sent_sent_rouge_l"]
    sent_sent_rouge_r = heterograph_data["sent_sent_rouge_r"]
    sent_sent_rouge_w = heterograph_data["sent_sent_rouge_w"]
    sent_sent_similarity_l = heterograph_data["sent_sent_similarity_l"]
    sent_sent_similarity_r = heterograph_data["sent_sent_similarity_r"]
    sent_sent_similarity_w = heterograph_data["sent_sent_similarity_w"]
    sent_token_contain_l = heterograph_data["sent_token_contain_l"]
    sent_token_contain_r = heterograph_data["sent_token_contain_r"]
    sent_token_contain_w = heterograph_data["sent_token_contain_w"]
    if not for_summary:
        doc_sent_contain_l = heterograph_data["doc_sent_contain_l"]
        doc_sent_contain_r = heterograph_data["doc_sent_contain_r"]
        doc_sent_contain_w = heterograph_data["doc_sent_contain_w"]
        doc_doc_similarity_l = heterograph_data["doc_doc_rouge_l"]
        doc_doc_similarity_r = heterograph_data["doc_doc_rouge_r"]
        doc_doc_similarity_w = heterograph_data["doc_doc_rouge_w"]

    graph = HeteroData()

    graph['token', 'similar_token', 'token'].edge_index = torch.tensor(
        [token_token_similarity_l, token_token_similarity_r])
    graph['token', 'similar_token', 'token'].edge_attr = torch.reshape(torch.tensor(token_token_similarity_w), (-1, 1))

    graph['token', 'followed_by', 'token'].edge_index = torch.tensor([token_token_follow_l, token_token_follow_r])
    graph['token', 'followed_by', 'token'].edge_attr = torch.reshape(torch.tensor(token_token_follow_w), (-1, 1))

    graph['sent', 'similar_sent_rouge', 'sent'].edge_index = torch.tensor(
        [sent_sent_rouge_l, sent_sent_rouge_r])
    graph['sent', 'similar_sent_rouge', 'sent'].edge_attr = torch.reshape(torch.tensor(sent_sent_rouge_w), (-1, 1))

    graph['sent', 'similar_sent_sbert', 'sent'].edge_index = torch.tensor(
        [sent_sent_similarity_l, sent_sent_similarity_r])
    graph['sent', 'similar_sent_sbert', 'sent'].edge_attr = torch.reshape(torch.tensor(sent_sent_similarity_w), (-1, 1))

    graph['sent', 'sent_contain_token', 'token'].edge_index = torch.tensor([sent_token_contain_l, sent_token_contain_r])
    graph['sent', 'sent_contain_token', 'token'].edge_attr = torch.reshape(torch.tensor(sent_token_contain_w), (-1, 1))

    if not for_summary:
        graph['doc', 'doc_contain_sent', 'sent'].edge_index = torch.tensor([doc_sent_contain_l, doc_sent_contain_r])
        graph['doc', 'doc_contain_sent', 'sent'].edge_attr = torch.reshape(torch.tensor(doc_sent_contain_w), (-1, 1))

        graph['doc', 'similar_doc_rouge', 'doc'].edge_index = torch.tensor([doc_doc_similarity_l, doc_doc_similarity_r])
        graph['doc', 'similar_doc_rouge', 'doc'].edge_attr = torch.reshape(torch.tensor(doc_doc_similarity_w), (-1, 1))

    return graph


if __name__ == "__main__":
    import jsonlines

    dataset_name = "multinews"
    token_type = "noun"
    samples = []
    with jsonlines.open("../data/%s_graph_%s_sentem.json" % (dataset_name, token_type)) as reader:
        for sample in reader:
            samples.append(sample)
            if len(samples)==8:
                break
    random.seed(42)
    samples = random.sample(samples, 3)
    for sample in samples:
        heterograph_source = build_graph(sample["heterograph_source"], for_summary=False)
        print("graph data for source documents")
        print(len(sample["heterograph_source"]["tokens_positions"]))
        print(len(sample["heterograph_source"]["sents_positions"]))
        print(len(sample["heterograph_source"]["docs_positions"]))
        print(heterograph_source)
        print(heterograph_source.metadata())
        print(heterograph_source.x_dict)

        heterograph_tgt = build_graph(sample["heterograph_tgt"], for_summary=True)
        print("graph data for target summary")
        print(len(sample["heterograph_tgt"]["tokens_positions"]))
        print(len(sample["heterograph_tgt"]["sents_positions"]))
        print(heterograph_tgt)
        print(heterograph_tgt.metadata())
        print(heterograph_tgt.x_dict)



