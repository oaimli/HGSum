from torch.utils.data import DataLoader, Dataset
import torch
import random
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
import sys

sys.path.append("../")
from graph.building_graph import build_graph


def concatenate_documents(all_docs, with_sent_sep, tokenizer, max_input_len):
    # dataset pre-processing
    for i, doc in enumerate(all_docs):
        doc = doc.replace("\n", " ").strip()
        doc = " ".join(doc.split())  # delete unnecessary space
        all_docs[i] = doc

    # concatenation of source documents
    max_doc_len = max_input_len // len(all_docs)
    tokenized_text = []
    for doc in all_docs:
        # truncate in advance
        doc_words = doc.split()
        if len(doc_words) > max_doc_len:
            doc = " ".join(doc_words[:max_doc_len])
        if with_sent_sep:
            sents = sent_tokenize(doc)
            doc = " ".join([sent + " <sent-sep>" for sent in sents])
        if len(all_docs) > 1:
            tokenized_text.extend(tokenizer.tokenize(doc)[:max_doc_len - 2])
        else:
            tokenized_text.extend(tokenizer.tokenize(doc)[:max_doc_len - 3])
        tokenized_text.append("<doc-sep>")
    tokenized_text = [tokenizer.bos_token] + tokenized_text + [tokenizer.eos_token]
    return tokenized_text


def tokenize_tgs(tgt, with_sent_sep, tokenizer, max_output_len):
    if with_sent_sep:
        sents = sent_tokenize(tgt)
        tgt = " ".join([sent + " <sent-sep>" for sent in sents])
    tokenized_text = tokenizer.tokenize(tgt)
    if max_output_len > 0:
        tokenized_text = tokenized_text[:max_output_len - 2]
    tokenized_text = [tokenizer.bos_token] + tokenized_text + [tokenizer.eos_token]
    return tokenized_text


class SummarizationDataset(Dataset):
    def __init__(
            self,
            dataset,
            dataset_name,
            with_sent_sep,
            tokenizer,
            max_input_len,
            max_output_len,
            mask_num=5,
            dataset_type="train",
    ):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.with_sent_sep = with_sent_sep
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.docsep_token_id = self.tokenizer.convert_tokens_to_ids("<doc-sep>")
        self.sentsep_token_id = self.tokenizer.convert_tokens_to_ids("<sent-sep>")
        self.mask_id = self.tokenizer.mask_token_id
        self.mask_num = mask_num
        self.dataset_type = dataset_type

        # print("loading Glove word embeddings")
        # self.glove = api.load("glove-wiki-gigaword-300")
        # print("loading SentenceTransformers")
        # self.sbert = SentenceTransformer("all-mpnet-base-v2", device='cpu')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset[idx]
        # single doc setting
        all_docs = entry["source_documents"]
        concatenated_source = concatenate_documents(all_docs, self.with_sent_sep, self.tokenizer, self.max_input_len)
        input_ids_source = self.tokenizer.convert_tokens_to_ids(concatenated_source)

        # # construct graph of source documents online
        # words_positions_source, sents_positions_source, docs_positions_source, heterograph_data = prepare_graph(concatenated_source, self.glove, online=True)

        # # load graph of source documents offline
        heterograph_data_source = entry["heterograph_source"]
        tokens_positions_source = torch.tensor(heterograph_data_source["tokens_positions"])
        sents_positions_source = torch.tensor(heterograph_data_source["sents_positions"])
        docs_positions_source = torch.tensor(heterograph_data_source["docs_positions"])
        heterograph_source = build_graph(heterograph_data=heterograph_data_source, for_summary=False)
        # print(heterograph_source)

        tgt = entry["summary"]
        tokenized_tgs = tokenize_tgs(tgt, False, self.tokenizer, self.max_output_len)
        output_ids = self.tokenizer.convert_tokens_to_ids(tokenized_tgs)

        input_ids_summary = self.tokenizer.convert_tokens_to_ids(tokenize_tgs(tgt, True, self.tokenizer, self.max_output_len))


        heterograph_data_tgt = entry["heterograph_tgt"]
        tokens_positions_tgt = torch.tensor(heterograph_data_tgt["tokens_positions"])
        sents_positions_tgt = torch.tensor(heterograph_data_tgt["sents_positions"])
        heterograph_tgt = build_graph(heterograph_data=heterograph_data_tgt, for_summary=True)
        # print(heterograph_tgt)

        # heterograph_source = None
        # words_positions_source = None
        # sents_positions_source = None
        # docs_positions_source = None
        # heterograph_tgt = None
        # words_positions_tgt = None
        # sents_positions_tgt = None

        if self.dataset_type == "train":
            return torch.tensor(input_ids_source), torch.tensor(
                output_ids), torch.tensor(input_ids_summary), heterograph_source, tokens_positions_source, sents_positions_source, docs_positions_source, heterograph_tgt, tokens_positions_tgt, sents_positions_tgt
        else:
            return torch.tensor(input_ids_source), torch.tensor(
                output_ids), torch.tensor(input_ids_summary), heterograph_source, tokens_positions_source, sents_positions_source, docs_positions_source, heterograph_tgt, tokens_positions_tgt, sents_positions_tgt, tgt


def collate_fn(batch):
    # A hack to know if this is bart or pegasus. DDP doesn't like global variables nor class-level memebr variables
    if batch[0][0][-1].item() == 2:
        pad_token_id = (
            1  # AutoTokenizer.from_pretrained('facebook/bart-base').pad_token_id
        )
    elif batch[0][0][-1].item() == 1:
        pad_token_id = (
            0  # AutoTokenizer.from_pretrained('google/pegasus-large').pad_token_id
        )
    else:
        assert False

    train = True
    if len(batch[0]) == 11:
        train = False
        tgt = [item[-1] for item in batch]
        batch = [item[:-1] for item in batch]
    input_ids_source, output_ids, input_ids_summary, heterograph_source, words_positions_source, sents_positions_source, docs_positions_source, heterograph_tgt, words_positions_tgt, sents_positions_tgt = list(
        zip(*batch))
    input_ids_source = torch.nn.utils.rnn.pad_sequence(
        input_ids_source, batch_first=True, padding_value=pad_token_id
    )
    output_ids = torch.nn.utils.rnn.pad_sequence(
        output_ids, batch_first=True, padding_value=pad_token_id
    )
    input_ids_summary = torch.nn.utils.rnn.pad_sequence(
        input_ids_summary, batch_first=True, padding_value=pad_token_id
    )
    if train:
        return input_ids_source, output_ids, input_ids_summary, heterograph_source, words_positions_source, sents_positions_source, docs_positions_source, heterograph_tgt, words_positions_tgt, sents_positions_tgt
    else:
        return input_ids_source, output_ids, input_ids_summary, heterograph_source, words_positions_source, sents_positions_source, docs_positions_source, heterograph_tgt, words_positions_tgt, sents_positions_tgt, tgt


def get_dataloader_summ(args, tokenizer, split_name, num_workers, is_shuffle):
    dataset_all = load_dataset('json', data_files=args.data_path + '%s_graph_noun_sentem.json' % args.dataset_name,
                               split='all')
    print("dataset all", len(dataset_all))

    random.seed(args.rand_seed)  # This is to control random selection of training and testing samples
    dataset = []
    if split_name == "train":
        dataset = dataset_all.filter(lambda s: s['label'] == 'train')
        print("dataset train all", len(dataset))
        if 0 < args.num_train_data < len(list(dataset)):
            dataset = dataset.select(random.choices(range(len(dataset)), k=args.num_train_data))
        print("dataset train selected", len(dataset))
    if split_name == "validation":
        dataset = dataset_all.filter(lambda s: s['label'] == 'val')
        print("dataset validation", len(dataset))
    if split_name == "test":
        dataset = dataset_all.filter(lambda s: s['label'] == 'test')
        if len(dataset) > args.num_test_data > 0:
            dataset = dataset.select(random.choices(range(len(dataset)), k=args.num_test_data))
        print("dataset test selected", len(dataset))

    summarization_dataset = SummarizationDataset(
        dataset=dataset,
        dataset_name=args.dataset_name,
        with_sent_sep=args.with_sent_sep,
        tokenizer=tokenizer,
        max_input_len=args.max_length_input,
        max_output_len=args.max_length_tgt,
        mask_num=args.mask_num,
        dataset_type=split_name,
    )

    return DataLoader(
        summarization_dataset,
        batch_size=args.batch_size,
        shuffle=is_shuffle,
        num_workers=num_workers,
        # sampler=sampler,
        collate_fn=collate_fn,
    )
