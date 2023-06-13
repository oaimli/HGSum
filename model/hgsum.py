import pandas as pd
import pdb
import json
import os
import argparse
import torch

from transformers import Adafactor, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from tokenization import LEDTokenizer
from modeling import LEDForConditionalGeneration
from dataloading import get_dataloader_summ

import sys

sys.path.append("../../")
from utils.metrics import rouge


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """
    This function is borrowed from fairseq.
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        count = (~pad_mask).sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        count = nll_loss.numel()

    nll_loss = nll_loss.sum() / count
    smooth_loss = smooth_loss.sum() / count
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss

    return loss, nll_loss


class HGSummarizer(pl.LightningModule):
    def __init__(self, args):
        super(HGSummarizer, self).__init__()
        self.args = args

        self.tokenizer = LEDTokenizer.from_pretrained(args.pretrained_primer)
        self.model = LEDForConditionalGeneration.from_pretrained(args.pretrained_primer)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.use_ddp = self.args.speed_strategy == "ddp"
        self.docsep_token_id = self.tokenizer.convert_tokens_to_ids("<doc-sep>")
        # or use eos_token as sent-sep
        self.tokenizer.add_tokens(["<sent-sep>"])
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.sentsep_token_id = self.tokenizer.convert_tokens_to_ids("<sent-sep>")

    def forward(self, input_ids_source, output_ids, input_ids_summary, heterograph_source, words_positions_source,
                sents_positions_source,
                docs_positions_source, heterograph_tgt, words_positions_tgt, sents_positions_tgt):
        device = input_ids_source.device
        # print("encoding source", len(heterograph_source))
        decoder_input_ids = output_ids[:, :-1]
        # get the input ids and attention masks together
        global_attention_mask_source = torch.zeros_like(input_ids_source).to(device)
        # put global attention on <s> token
        global_attention_mask_source[:, 0] = 1
        # put global attention on <doc-sep> token
        global_attention_mask_source[input_ids_source == self.docsep_token_id] = 1
        # put global attention on <sent-sep> token
        global_attention_mask_source[input_ids_source == self.sentsep_token_id] = 1

        attention_mask = torch.ones_like(input_ids_source).to(device)
        attention_mask = attention_mask.type_as(input_ids_source)
        attention_mask[input_ids_source == self.pad_token_id] = 0
        # encoding source documents
        outputs_source = self.model(
            input_ids_source,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            global_attention_mask=global_attention_mask_source,
            use_cache=False,
            heterograph=heterograph_source,
            words_positions_source=words_positions_source,
            sents_positions_source=sents_positions_source,
            docs_positions_source=docs_positions_source
        )
        lm_logits = outputs_source.logits
        assert lm_logits.shape[-1] == self.model.config.vocab_size

        # print("encoding summary", input_ids_summary.size())
        # pdb.set_trace()
        # encoding summary
        # get the input ids and attention masks together
        global_attention_mask_summary = torch.zeros_like(input_ids_summary).to(device)
        # put global attention on <s> token
        global_attention_mask_summary[:, 0] = 1
        # put global attention on <sent-sep> token
        global_attention_mask_summary[input_ids_summary == self.sentsep_token_id] = 1
        outputs_summary = self.model(
            input_ids_summary.to(device),
            decoder_input_ids=None,
            global_attention_mask=global_attention_mask_summary,
            use_cache=False,
            heterograph=heterograph_tgt,
            words_positions_source=words_positions_tgt,
            sents_positions_source=sents_positions_tgt,
            docs_positions_source=None
        )

        return lm_logits, outputs_source.mgat_outputs, outputs_source.sagpooling_outputs, outputs_summary.mgat_outputs
        # return lm_logits, outputs_source.mgat_outputs, outputs_source.sagpooling_outputs, None

    def configure_optimizers(self):
        if self.args.adafactor:
            optimizer = Adafactor(
                self.parameters(),
                lr=self.args.lr,
                scale_parameter=False,
                relative_step=False,
            )
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=self.args.warmup_steps
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.args.total_steps,
            )
        if self.args.fix_lr:
            return optimizer
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def shared_step(self, input_ids_source, output_ids, input_ids_summary, heterograph_source, words_positions_source,
                    sents_positions_source,
                    docs_positions_source, heterograph_tgt, words_positions_tgt, sents_positions_tgt):
        lm_logits, mgat_outputs_source, sagpooling_ouputs, mgat_outputs_summary = self.forward(input_ids_source,
                                                                                               output_ids,
                                                                                               input_ids_summary,
                                                                                               heterograph_source,
                                                                                               words_positions_source,
                                                                                               sents_positions_source,
                                                                                               docs_positions_source,
                                                                                               heterograph_tgt,
                                                                                               words_positions_tgt,
                                                                                               sents_positions_tgt)

        # print("shared step")
        # graph similarity loss
        cos = torch.nn.CosineSimilarity(dim=1)
        graph_sim = torch.mean(cos(sagpooling_ouputs, mgat_outputs_summary))
        # coss-entropy loss
        labels = output_ids[:, 1:].clone()

        if self.args.label_smoothing == 0.0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs,
                labels,
                self.args.label_smoothing,
                ignore_index=self.pad_token_id,
            )
        if torch.isnan(loss):
            pdb.set_trace()

        return 0.5 * loss + 0.5 * graph_sim

    def training_step(self, batch, batch_idx):
        input_ids_source, output_ids, input_ids_summary, heterograph_source, words_positions_source, sents_positions_source, docs_positions_source, heterograph_tgt, words_positions_tgt, sents_positions_tgt = batch
        loss = self.shared_step(input_ids_source, output_ids, input_ids_summary, heterograph_source,
                                words_positions_source,
                                sents_positions_source, docs_positions_source, heterograph_tgt, words_positions_tgt,
                                sents_positions_tgt)
        # print("training step")

        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]["lr"]
        tensorboard_logs = {
            "train_loss": loss,
            "lr": lr,
            "input_size_source": input_ids_source.numel(),
            "output_size": output_ids.numel(),
            "mem": torch.cuda.memory_allocated(loss.device) / 1024 ** 3
            if torch.cuda.is_available()
            else 0,
        }
        self.logger.log_metrics(tensorboard_logs, step=self.global_step)
        return loss

    def compute_rouge_batch(self, input_ids, gold_str, heterograph_source, words_positions_source,
                            sents_positions_source, docs_positions_source, batch_idx):
        # get the input ids and attention masks together
        global_attention_mask = torch.zeros_like(input_ids).cuda()
        # put global attention on <s> token
        global_attention_mask[:, 0] = 1
        global_attention_mask[input_ids == self.docsep_token_id] = 1
        global_attention_mask[input_ids == self.sentsep_token_id] = 1
        attention_mask = torch.ones_like(input_ids).cuda()
        attention_mask[input_ids == self.pad_token_id] = 0
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            use_cache=False,
            max_length=self.args.max_length_tgt,
            min_length=self.args.min_length_tgt,
            num_beams=self.args.beam_size,
            no_repeat_ngram_size=3 if self.args.apply_triblck else None,
            length_penalty=self.args.length_penalty,
            heterograph=heterograph_source,
            words_positions_source=words_positions_source,
            sents_positions_source=sents_positions_source,
            docs_positions_source=docs_positions_source,

        )

        generated_str = self.tokenizer.batch_decode(
            generated_ids.tolist(), skip_special_tokens=True
        )

        if self.args.mode == "test":
            if self.args.apply_triblck:
                output_dir = os.path.join(
                    self.args.model_path,
                    "generated_txt_%d_%s_triblck_beam=%d_%d_%d"
                    % (
                        self.args.mask_num,
                        self.args.dataset_name,
                        self.args.beam_size,
                        self.args.max_length_input,
                        self.args.max_length_tgt,
                    ),
                )
            else:
                output_dir = os.path.join(
                    self.args.model_path,
                    "generated_txt_%d_%s_beam=%d_%d_%d"
                    % (
                        self.args.mask_num,
                        self.args.dataset_name,
                        self.args.beam_size,
                        self.args.max_length_input,
                        self.args.max_length_tgt,
                    ),
                )
            if batch_idx == 0:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                else:
                    for file in os.listdir(output_dir):
                        os.remove(os.path.join(output_dir, file))
            idx = len(os.listdir(output_dir))
        result_batch = []
        if self.args.debug_mode:
            pdb.set_trace()
        for ref, pred in zip(gold_str, generated_str):
            # change <n> to \n
            pred = pred.replace("<n>", "\n")

            if self.args.mode == "test":
                with open(os.path.join(output_dir, "%d.txt" % (idx)), "w") as of:
                    of.write(pred)
                idx += 1

            s = rouge(reference=ref, candidate=pred, use_stemmer=True,
                      types=["rouge1", "rouge2", "rougeL", "rougeLsum"], split_summaries=True)
            result_batch.append(
                (
                    s["rouge1"]["recall"],
                    s["rouge1"]["precision"],
                    s["rouge1"]["fmeasure"],
                    s["rouge2"]["recall"],
                    s["rouge2"]["precision"],
                    s["rouge2"]["fmeasure"],
                    s["rougeL"]["recall"],
                    s["rougeL"]["precision"],
                    s["rougeL"]["fmeasure"],
                    s["rougeLsum"]["recall"],
                    s["rougeLsum"]["precision"],
                    s["rougeLsum"]["fmeasure"]
                )
            )
        return result_batch

    def validation_step(self, batch, batch_idx):
        for p in self.model.parameters():
            p.requires_grad = False
        input_ids_source, output_ids, input_ids_summary, heterograph_source, words_positions_source, sents_positions_source, docs_positions_source, heterograph_tgt, words_positions_tgt, sents_positions_tgt, tgt = batch
        loss = self.shared_step(input_ids_source, output_ids, input_ids_summary, heterograph_source,
                                words_positions_source,
                                sents_positions_source, docs_positions_source, heterograph_tgt, words_positions_tgt,
                                sents_positions_tgt)
        # print("validation step", input_ids_source.size())
        if self.args.compute_rouge:
            result_batch = self.compute_rouge_batch(input_ids_source, tgt, heterograph_source, words_positions_source,
                                                    sents_positions_source, docs_positions_source, batch_idx)
            return {"vloss": loss, "rouge_result": result_batch}
        else:
            return {"vloss": loss}

    def compute_rouge_all(self, outputs, output_file=None):
        rouge_result_all = [r for b in outputs for r in b["rouge_result"]]
        names = []
        for rouge in ["1", "2", "L", "Lsum"]:
            names.extend(
                [
                    "rouge-{}-r".format(rouge),
                    "rouge-{}-p".format(rouge),
                    "rouge-{}-f".format(rouge),
                ]
            )
        rouge_results = pd.DataFrame(rouge_result_all, columns=names)
        avg = [rouge_results[c].mean() for c in rouge_results.columns]
        rouge_results.loc["avg_score"] = avg
        if output_file:
            csv_name = (
                    self.args.model_path
                    + output_file
                    + "-%d.csv" % (torch.distributed.get_rank() if self.use_ddp else 0)
            )
            rouge_results.to_csv(csv_name)

        avgf = (avg[2] + avg[5] + avg[11]) / 3
        metrics = avg
        print("Validation Result at Step %d" % (self.global_step))
        print(
            "Rouge-1 r score: %f, Rouge-1 p score: %f, Rouge-1 f-score: %f"
            % (metrics[0], metrics[1], metrics[2])
        )
        print(
            "Rouge-2 r score: %f, Rouge-2 p score: %f, Rouge-2 f-score: %f"
            % (metrics[3], metrics[4], metrics[5])
        )
        print(
            "Rouge-L r score: %f, Rouge-L p score: %f, Rouge-L f-score: %f"
            % (metrics[6], metrics[7], metrics[8])
        )
        print(
            "Rouge-Lsum r score: %f, Rouge-Lsum p score: %f, Rouge-Lsum f-score: %f"
            % (metrics[9], metrics[10], metrics[11])
        )
        return names, metrics, avgf

    def validation_epoch_end(self, outputs):
        for p in self.model.parameters():
            p.requires_grad = True

        vloss = torch.stack([x["vloss"] for x in outputs]).mean()
        self.log("vloss", vloss, sync_dist=True if self.use_ddp else False)
        if self.args.compute_rouge:
            names, metrics, avgf = self.compute_rouge_all(outputs, output_file="valid")
            metrics = [vloss] + metrics
            names = ["vloss"] + names
            logs = dict(zip(*[names, metrics]))
            self.logger.log_metrics(logs, step=self.global_step)
            self.log("avgf", avgf)
            return {
                "avg_val_loss": vloss,
                "avgf": avgf,
                "log": logs,
                "progress_bar": logs,
            }
        else:
            logs = {"vloss": vloss}
            self.logger.log_metrics(logs, step=self.global_step)
            return {"vloss": vloss, "log": logs, "progress_bar": logs}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        tloss = torch.stack([x["vloss"] for x in outputs]).mean()
        self.log("tloss", tloss, sync_dist=True if self.use_ddp else False)
        output_file = "test_%s_%d_%d_beam=%d_lenPen=%.2f" % (
            self.args.dataset_name,
            self.args.max_length_input,
            self.args.max_length_tgt,
            self.args.beam_size,
            self.args.length_penalty,
        )
        output_file = (
            output_file
            + "_fewshot_%d_%d" % (self.args.num_train_data, self.args.rand_seed)
            if self.args.fewshot
            else output_file
        )
        names, metrics, avgf = self.compute_rouge_all(outputs, output_file=output_file)
        metrics = [tloss, avgf] + metrics
        names = ["tloss", "avgf"] + names
        logs = dict(zip(*[names, metrics]))
        self.logger.log_metrics(logs, step=self.global_step)
        self.log("avgf", avgf)
        # self.log_dict(logs)
        return {"avg_test_loss": tloss, "avgf": avgf, "log": logs, "progress_bar": logs}


def train(args):
    model = HGSummarizer(args)

    # load dataset
    train_dataloader = get_dataloader_summ(args, model.tokenizer, 'train', args.num_workers, True)
    valid_dataloader = get_dataloader_summ(args, model.tokenizer, 'validation', args.num_workers, False)

    # initialize checkpoint
    if args.ckpt_path is None:
        args.ckpt_path = os.path.join(args.model_path, "summ_checkpoints/")

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_path,
        filename="{step}-{vloss:.2f}-{avgf:.4f}",
        save_top_k=args.save_top_k,
        monitor="avgf",
        mode="max",
        save_on_train_epoch_end=False,
    )
    early_stopping = EarlyStopping(monitor='vloss', patience=3, mode='min')

    # initialize logger
    logger = TensorBoardLogger(args.model_path + "tb_logs", name=args.model_name)

    # initialize trainer
    trainer = pl.Trainer(
        devices=args.gpus,
        accelerator=args.accelerator,
        auto_select_gpus=True,
        strategy=args.speed_strategy,
        track_grad_norm=-1,
        max_steps=args.total_steps * args.accum_batch,
        replace_sampler_ddp=False,
        accumulate_grad_batches=args.accum_batch,
        # val_check_interval=0.5,
        check_val_every_n_epoch=1 if args.num_train_data > 100 else 5,
        logger=logger,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback, early_stopping],
        precision=32,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches else 1.0,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches else 1.0,
        num_sanity_val_steps=0
    )

    # pdb.set_trace()
    trainer.fit(model, train_dataloader, valid_dataloader)

    if args.test_imediate:
        args.resume_ckpt = checkpoint_callback.best_model_path
        print(args.resume_ckpt)
        if args.test_batch_size != -1:
            args.batch_size = args.test_batch_size
        args.mode = "test"
        test(args)


def test(args):
    # initialize trainer
    trainer = pl.Trainer(
        devices=1,
        auto_select_gpus=True,
        accelerator=args.accelerator,
        track_grad_norm=-1,
        max_steps=args.total_steps * args.accum_batch,
        replace_sampler_ddp=False,
        log_every_n_steps=5,
        precision=32,
        limit_test_batches=args.limit_test_batches if args.limit_test_batches else 1.0
    )

    if args.resume_ckpt is not None:
        model = HGSummarizer.load_from_checkpoint(args.resume_ckpt, args=args)
    else:
        model = HGSummarizer(args)

    # load dataset
    test_dataloader = get_dataloader_summ(args, model.tokenizer, 'test', args.num_workers, False)

    # test
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    seed_everything(42, workers=True)
    parser = argparse.ArgumentParser()

    # Gneral
    parser.add_argument("--gpus", default=1, type=int, help="The number of gpus to use")
    parser.add_argument("--accelerator", default="gpu", type=str, choices=["gpu", "cpu"])
    parser.add_argument("--speed_strategy", default=None, type=str, help="Accelerator strategy, e.g., ddp")
    parser.add_argument("--mode", default="train", choices=["train", "test"])
    parser.add_argument("--model_name", default="HGSum")
    parser.add_argument("--pretrained_primer", type=str, default=None,
                        help="Name or path of pretrained PRIMERA from Huggingface, or the model to be tested")
    parser.add_argument("--with_sent_sep", action="store_true",
                        help="Insert <sent-sep> at the end of each sentence when concatenating different documents")
    parser.add_argument("--debug_mode", action="store_true", help="Set true if to debug")
    parser.add_argument("--compute_rouge", action="store_true", help="whether to compute rouge in validation steps")
    parser.add_argument("--progress_bar_refresh_rate", default=1, type=int)
    parser.add_argument("--model_path", type=str, default="result/",
                        help="The path to save output and checkpoints in training and testing")
    parser.add_argument("--ckpt_path", type=str, default=None, help="dir to save checkpoints")
    parser.add_argument("--save_top_k", default=3, type=int)
    parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from", default=None)
    parser.add_argument("--data_path", type=str, default="../../datasets/")
    parser.add_argument("--dataset_name", type=str, default="multinews",
                        choices=["multinews", "arxiv", "multixscience", "wcep_10", "wcep_100", "peersum_r",
                                 "peersum_rc", "peersum_all"])
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers to use for dataloader")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--max_length_input", default=4096, type=int)
    parser.add_argument("--max_length_tgt", default=1024, type=int)
    parser.add_argument("--min_length_tgt", default=0, type=int)
    parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
    parser.add_argument("--adafactor", action="store_true", help="Use adafactor optimizer")
    parser.add_argument("--grad_ckpt", action="store_true", help="Enable gradient checkpointing to save memory")
    parser.add_argument("--rand_seed", type=int, default=42,
                        help="Seed for random sampling, useful for few shot learning")

    # For training
    parser.add_argument("--limit_train_batches", type=float, default=1.0, help="Use limited batches in training")
    parser.add_argument("--limit_val_batches", type=float, default=1.0, help="Use limited batches in validation")
    parser.add_argument("--lr", type=float, default=3e-5, help="Maximum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--accum_data_per_step", type=int, default=16, help="Number of data per step")
    parser.add_argument("--total_steps", type=int, default=50000, help="Number of steps to train")
    parser.add_argument("--num_train_data", type=int, default=-1,
                        help="Number of training data, -1 for full dataset and any positive number indicates how many data to use")
    parser.add_argument("--fix_lr", action="store_true", help="use fix learning rate")
    parser.add_argument("--test_imediate", action="store_true", help="test on the best checkpoint")
    parser.add_argument("--fewshot", action="store_true", help="whether this is a run for few shot learning")

    # For testing
    parser.add_argument("--limit_test_batches", type=float, default=1.0,
                        help="Number of batches to test in the test mode")
    parser.add_argument("--beam_size", type=int, default=1, help="size of beam search")
    parser.add_argument("--length_penalty", type=float, default=1, help="length penalty of generated text")
    parser.add_argument("--mask_num", type=int, default=0, help="Number of masks in the input of summarization data")
    parser.add_argument("--test_batch_size", type=int, default=-1,
                        help="batch size for test, used in few shot evaluation.")
    parser.add_argument("--apply_triblck", action="store_true",
                        help="whether apply trigram block in the evaluation phase")
    parser.add_argument("--num_test_data", type=int, default=-1, help="the number of testing data")

    args = parser.parse_args()
    args.accum_batch = args.accum_data_per_step // args.batch_size

    if args.gpus > 0:
        args.accelerator = "gpu"
    else:
        args.accelerator = "cpu"
        args.gpus = 1
    if args.num_workers == -1:
        args.num_workers = os.cpu_count()
    print(args)

    if not os.path.exists(args.model_path):  # this is used to save the checkpoints and logs
        os.makedirs(args.model_path)
    with open(os.path.join(args.model_path, "args_%s_%s.json" % (args.mode, args.dataset_name)), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    if args.mode == "train":
        train(args)
    if args.mode == "test":
        test(args)
