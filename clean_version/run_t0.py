import logging
import sys

import datasets
import numpy as np
import torch


try:
    load_metric = datasets.load_metric
except AttributeError:  # for newer datasets versions
    import evaluate

    load_metric = evaluate.load


from data_collator import ListDataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model
from trainer import TTTTrainer
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from ttt.dataloader import DatasetByPrompt, TTTEvalDataset, TTTOfflineLoopDataset
from ttt.options import DataArguments, ModelArguments, TestArguments
from ttt.utils import compute_metrics


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, TestArguments))
    model_args, data_args, training_args, test_args = parser.parse_args_into_dataclasses()

    # expand the model path to the full hub identifier expected by HuggingFace
    model_args.model_name_or_path = "bigscience/" + model_args.model_name_or_path

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    for k, v in vars(test_args).items():
        if not hasattr(config, k):
            setattr(config, k, v)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, config=config)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    if test_args.peft_option == "lora":
        lora_config = LoraConfig(
            r=test_args.bottleneck_dim,
            lora_alpha=test_args.lora_alpha,
            lora_dropout=test_args.lora_dropout,
            bias="none",
            target_modules=["q", "v", "k", "o", "wi", "wo"],
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    elif test_args.peft_option == "bitfit":
        for name, param in model.named_parameters():
            param.requires_grad = "bias" in name
    elif test_args.peft_option != "full":
        for param in model.parameters():
            param.requires_grad = False

    model.resize_token_embeddings(len(tokenizer))

    dataset = DatasetByPrompt(data_args, model_args.cache_dir, tokenizer)
    train_dataset = TTTOfflineLoopDataset(
        dataset, test_args, test_args.train_random_n_prompts, training_args.per_device_eval_batch_size
    )
    eval_dataset = TTTEvalDataset(dataset)

    expand = test_args.loss_option in ["consistency", "pseudo_train", "consistency_pseudo_train"]
    data_collator = ListDataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, expand_list=expand)

    metrics = load_metric(
        data_args.dataset_name if test_args.metric_name == "none" else test_args.metric_name,
        data_args.subset_name if test_args.metric_name == "none" else None,
        cache_dir=model_args.cache_dir,
    )

    trainer = TTTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    def make_compute_metrics(dataset, prefix):
        def _compute(eval_pred):
            if isinstance(eval_pred.predictions, (list, tuple)):
                flat = []
                for p in eval_pred.predictions:
                    arr = np.asarray(p)
                    flat.append(arr.reshape(-1))
                logprobs = np.concatenate(flat)
            else:
                logprobs = np.asarray(eval_pred.predictions).reshape(-1)
            step = trainer.state.global_step
            num_instances = getattr(dataset, "num_instances", len(dataset))
            results, _ = compute_metrics(
                logprobs,
                num_instances,
                dataset.num_choices,
                dataset.num_prompts,
                dataset.gold_labels,
                metrics=metrics,
                fout_name=training_args.output_dir,
                suffix=f"{prefix}_{step}",
            )
            return results

        return _compute

    trainer.compute_metrics = make_compute_metrics(eval_dataset, "eval")
    trainer.compute_train_metrics = make_compute_metrics(train_dataset, "train")

    trainer.data_collator = data_collator

    trainer.train_ttt()
    trainer.evaluate()


if __name__ == "__main__":
    main()
