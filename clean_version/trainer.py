import math
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from transformers.optimization import Adafactor, get_scheduler
from transformers.trainer import Trainer, get_parameter_names
from ttt.dataloader import TTTOfflineLoopDataset


class TTTTrainer(Trainer):
    """Trainer with parameter-efficient optimizer setup and custom loss."""

    def __init__(
        self,
        *args,
        compute_unsupervised_metrics: Optional[Any] = None,
        **kwargs,
    ):
        """Extend ``Trainer`` to keep track of unsupervised metric callback."""
        self.compute_unsupervised_metrics = compute_unsupervised_metrics
        if "compute_unsupervised_metrics" in kwargs:
            kwargs.pop("compute_unsupervised_metrics")
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [n for n in decay_parameters if "bias" not in n]
        decay_group = [p for n, p in self.model.named_parameters() if n in decay_parameters and p.requires_grad]
        non_decay_group = [
            p for n, p in self.model.named_parameters() if n not in decay_parameters and p.requires_grad
        ]
        optimizer_grouped_parameters = []
        if len(decay_group) > 0:
            optimizer_grouped_parameters.append({"params": decay_group, "weight_decay": self.args.weight_decay})
        if len(non_decay_group) > 0:
            optimizer_grouped_parameters.append({"params": non_decay_group, "weight_decay": 0.0})
        optimizer_cls = Adafactor if self.args.adafactor else AdamW
        optimizer_kwargs = {"lr": self.args.learning_rate}
        if not self.args.adafactor:
            optimizer_kwargs.update(
                {"betas": (self.args.adam_beta1, self.args.adam_beta2), "eps": self.args.adam_epsilon}
            )
        else:
            optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        return self.lr_scheduler

    def train_ttt(self, *args, **kwargs):
        # simplified wrapper around `train`
        return super().train(*args, **kwargs)

    def _merge_inputs(self, inputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Pad and concatenate a list of feature dictionaries."""
        merged: Dict[str, torch.Tensor] = {}
        for k in inputs[0].keys():
            values = [dic[k] for dic in inputs]
            if values[0].dim() == 2:
                # Manually pad along sequence length since ``pad_sequence``
                # expects (seq_len, feature) tensors.
                pad_token_id = getattr(self.processing_class, "pad_token_id", None)
                pad_value = -100 if k == "labels" else pad_token_id or 0
                max_len = max(t.size(1) for t in values)
                padded = []
                for t in values:
                    if t.size(1) < max_len:
                        pad = t.new_full((t.size(0), max_len - t.size(1)), pad_value)
                        t = torch.cat([t, pad], dim=1)
                    padded.append(t)
                merged[k] = torch.cat(padded, dim=0)
            else:
                merged[k] = torch.cat(values, dim=0)
        return merged

    def _compute_consistency_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mask = labels != -100
        logits = logits * mask.unsqueeze(2)
        bsz, length, vsz = logits.size()
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        instance_bsz = 1
        random_n_prompts = bsz
        log_probs = log_probs.view(instance_bsz, -1, length, vsz)
        bsz = log_probs.size(0)
        avg = torch.logsumexp(log_probs, dim=1) - math.log(random_n_prompts)
        total_tokens = mask.sum().float()
        if self.model.config.jsd:
            if self.model.config.detach_kl_left:
                log_probs = log_probs.detach()
            if self.model.config.detach_kl_right:
                avg = avg.detach()
            avg = avg.unsqueeze(1).expand(bsz, random_n_prompts, length, vsz)
            loss = torch.nn.functional.kl_div(avg, log_probs, reduction="sum", log_target=True)
        else:
            if self.model.config.detach_kl_left:
                avg = avg.detach()
            if self.model.config.detach_kl_right:
                log_probs = log_probs.detach()
            avg = avg.unsqueeze(1).expand(bsz, random_n_prompts, length, vsz)
            loss = torch.nn.functional.kl_div(log_probs, avg, reduction="sum", log_target=True) / total_tokens
        return loss

    def _compute_entropy_loss(self, logprobs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mask = labels != -100
        num_targets = self.model.config.num_choices
        logprobs = -logprobs.view(labels.size())
        loss = (logprobs * mask).sum(1)
        probs = loss.exp().view(-1, num_targets)
        probs = torch.pow(probs, self.model.config.prob_temperature)
        normalized = probs / probs.sum(1, keepdims=True)
        rand_n = (
            self.model.config.train_random_n_prompts
            if getattr(self.model.config, "train_random_n_prompts", -1) > 0
            else normalized.size(0)
        )
        normalized = normalized.view(-1, rand_n, normalized.size(-1))
        if self.model.config.combine_option == "uniform":
            marginal = normalized.mean(1)
        else:
            ents = -(normalized * normalized.log()).sum(-1)
            ents = ents / ents.sum(1, keepdims=True)
            marginal = (normalized * ents.unsqueeze(-1)).sum(1)
        return -(marginal * marginal.log()).sum(1).mean()

    def _compute_token_level_entropy_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mask = labels != -100
        bsz, length, vsz = logits.size()
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        instance_bsz = 1
        random_n_prompts = bsz
        log_probs = log_probs.view(instance_bsz, -1, length, vsz)
        avg = torch.logsumexp(log_probs, dim=1) - math.log(random_n_prompts)
        mask_first = mask[0:bsz:random_n_prompts]
        loss = -(avg.exp() * avg).sum(-1) * mask_first
        total_tokens = mask.sum().float()
        return loss.sum() / total_tokens

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if isinstance(inputs, list):
            inputs = self._merge_inputs(inputs)

        labels = inputs.get("labels")
        outputs = model(**inputs, output_hidden_states=False, return_dict=True)

        if labels is not None and getattr(model.config, "test_mode", None) == "ttt_t0" and model.training:
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            ce_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            nll_loss = (ce_loss.view(labels.size()) * (labels != -100)).sum(1)

            option = getattr(model.config, "loss_option", "none")
            if option == "entropy":
                total_loss = self._compute_entropy_loss(ce_loss, labels)
            elif option == "consistency":
                total_loss = self._compute_consistency_loss(logits, labels)
            elif option == "consistency_pseudo_train":
                total_loss = (
                    self._compute_consistency_loss(logits, labels) + model.config.pseudo_train_loss_weight * nll_loss
                )
            elif option == "pseudo_train":
                total_loss = nll_loss
            elif option == "token_level_entropy":
                total_loss = self._compute_token_level_entropy_loss(logits, labels)
            else:
                raise ValueError(f"Unknown loss option: {option}")
            loss = total_loss
        elif labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            ce_loss = loss_fct(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
            nll = (ce_loss.view(labels.size()) * (labels != -100)).sum(1)
            if not model.training and getattr(model.config, "test_mode", None) == "ttt_t0":
                # During evaluation expose log-probabilities via logits
                outputs.logits = -nll.detach()
                loss = nll.mean()
            else:
                loss = nll
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset: Optional[Union[List[Dict[str, torch.Tensor]], torch.utils.data.Dataset]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """Evaluate on validation and, optionally, training sets."""
        metrics = super().evaluate(
            eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        if (
            metric_key_prefix == "eval"
            and self.train_dataset is not None
            and self.compute_metrics is not None
            and hasattr(self, "compute_train_metrics")
        ):
            orig_compute = self.compute_metrics
            self.compute_metrics = self.compute_train_metrics
            try:
                train_metrics = super().evaluate(
                    eval_dataset=self.train_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix="train",
                )
            finally:
                self.compute_metrics = orig_compute
            metrics.update({f"train_{k}": v for k, v in train_metrics.items()})
        return metrics

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """Override to use batch size 1 when evaluating ``TTTOfflineLoopDataset``."""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if isinstance(eval_dataset, TTTOfflineLoopDataset):
            orig = self.args.per_device_eval_batch_size
            self.args.per_device_eval_batch_size = 1
            try:
                loader = super().get_eval_dataloader(eval_dataset)
            finally:
                self.args.per_device_eval_batch_size = orig
            return loader
        return super().get_eval_dataloader(eval_dataset)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        """Merge nested batches before forwarding to the base implementation."""
        if isinstance(inputs, list):
            inputs = self._merge_inputs(inputs)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
