import numpy as np
import scipy
import math
import os
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def index_median(array):
    x = np.argsort(array)
    h = len(x) // 2
    # return (x[h] + x[h+1]) // 2 if len(x) % 2 == 0 else x[h]
    return h


def write_results_to_file(
    fout_name,
    suffix,
    all_prompt_metrics,
    all_prompt_predictions,
    avg_ensemble_metrics,
    avg_ensemble_preds,
    vote_ensemble_metrics,
    vote_ensemble_preds,
    golds,
    avg_entropy=None,
    extra_results: Optional[Dict[str, Any]] = None,
):
    """Write evaluation metrics to multiple files.

    Each metric gets its own output file (``<metric>_<suffix>``) containing
    the full set of aggregated results. The per-example statistics are written
    with the correct metric label instead of the hard-coded ``acc`` prefix.
    ``extra_results`` can be used to inject additional fields such as POSIX
    before the metrics are written.
    """

    results: Dict[str, Any] = {}
    per_metric_data: Dict[str, Any] = {}
    for k in all_prompt_metrics[0].keys():
        all_metrics = [pptm[k] * 100 for pptm in all_prompt_metrics]
        median_prompt = all_prompt_predictions[index_median(all_metrics)]
        max_prompt = all_prompt_predictions[np.argsort(all_metrics)[-1]]
        results["max_" + k] = round(np.max(all_metrics), 2)
        results["median_" + k] = round(np.median(all_metrics), 2)
        results["mean_" + k] = round(np.mean(all_metrics), 2)
        results["min_" + k] = round(np.min(all_metrics), 2)
        results["std_" + k] = round(np.std(all_metrics), 2)
        results["avg_ensemble_" + k] = round(avg_ensemble_metrics[k] * 100, 2)
        results["vote_ensemble_" + k] = round(vote_ensemble_metrics[k] * 100, 2)
        per_metric_data[k] = (all_metrics, median_prompt, max_prompt)

    # compute spread metrics
    if "max_precision" in results and "min_precision" in results:
        results["precision_spread"] = round(results["max_precision"] - results["min_precision"], 2)
    if "max_recall" in results and "min_recall" in results:
        results["recall_spread"] = round(results["max_recall"] - results["min_recall"], 2)
    if "max_f1" in results and "min_f1" in results:
        results["f1_spread"] = round(results["max_f1"] - results["min_f1"], 2)

    try:
        num_labels = int(max(max(preds) for preds in all_prompt_predictions)) + 1
        fk, po = _fleiss_kappa_po(all_prompt_predictions, num_labels)
        results["fleiss_kappa"] = round(float(fk), 4)
        results["raw_agreement"] = round(float(po), 4)
    except Exception as e:
        logger.warning("Fleiss kappa computation failed: %s", str(e))

    if extra_results:
        results.update(extra_results)

    for metric, (metric_values, median_prompt, max_prompt) in per_metric_data.items():
        if fout_name.startswith("results"):
            nfout = fout_name + f".{metric}_{suffix}"
        else:
            nfout = os.path.join(fout_name, f"{metric}_{suffix}")
        with open(nfout, "w") as fout:
            fout.write(",".join([f"{kk}={vv}" for kk, vv in results.items()]) + "\n")
            if avg_entropy is not None:
                fout.write(f"{metric}: " + " ".join([str(vv) for vv in metric_values]) + "\n")
                fout.write("ent: " + " ".join([str(vv) for vv in avg_entropy]) + "\n")
            for ii in range(len(all_prompt_predictions[0])):
                s = (
                    ",".join(
                        [
                            f"gold={golds[ii]}",
                            f"median={median_prompt[ii]}",
                            f"max={max_prompt[ii]}",
                            f"avg_esemb={avg_ensemble_preds[ii]}",
                            f"vote_esemb={vote_ensemble_preds[ii]}",
                        ]
                    )
                    + ","
                )
                s += " ".join([
                    str(all_prompt_predictions[jj][ii])
                    for jj in range(len(all_prompt_predictions))
                ])
                fout.write(s + "\n")

    return results


def write_unsupervised_results_to_file(fout_name, results, all_prompt_predictions, golds=None):
    with open(fout_name, "w") as fout:
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                fout.write("{}={}".format(key, " ".join([str(kk) for kk in value])) + "\n")
            else:
                fout.write("{}={}".format(key, value) + "\n")

        # output predictions of prompts for each example
        for ii in range(len(all_prompt_predictions[0])):
            s = f"gold={golds[ii]}, " if golds is not None else ""
            s += " ".join([str(all_prompt_predictions[jj][ii]) for jj in range(len(all_prompt_predictions))])
            fout.write(s + "\n")


def compute_metrics(
    logprobs,
    num_examples,
    num_targets,
    num_prompts,
    golds=None,
    metrics=None,
    fout_name=None,
    suffix=None,
    pseudo_dist="smooth",
    return_all_prompt_preds=False,
    random_selection_ensemble=0.0,
    self_train=False,
    **kwargs
):
    predictions = [[] for _ in range(num_prompts)]
    entropies = [[] for _ in range(num_prompts)]
    avg_ensemble_predictions = []
    vote_ensemble_predictions = []
    all_avg_probs = []  # only used when num of examples=1
    posix_values = []
    idx = 0
    logits = [[] for _ in range(num_prompts)]
    for eidx in range(num_examples):
        avg_probs = np.zeros(num_targets)
        example_logits = []
        for pidx in range(num_prompts):
            max_ll, pred_label = -np.inf, -1
            # actually, the number of labels of each prompt should be the same
            normalized_probs = np.zeros(num_targets)
            logit = []
            for ii in range(num_targets):
                if logprobs[idx] > max_ll:
                    max_ll, pred_label = logprobs[idx], ii
                normalized_probs[ii] = math.exp(logprobs[idx])
                logit.append(logprobs[idx])
                idx += 1
            logits[pidx].append(logit)
            example_logits.append(logit)
            normalized_probs = normalized_probs / normalized_probs.sum()
            entropies[pidx].append(-(normalized_probs * np.log(normalized_probs)).sum())
            avg_probs += normalized_probs
            all_avg_probs.append(normalized_probs)
            predictions[pidx].append(pred_label)

        # import pdb; pdb.set_trace()
        posix_values.append(_posix_from_logits(example_logits))

        if 0.0 < random_selection_ensemble < 1.0 and num_examples == 1:
            selected_prompts = np.random.permutation(num_prompts)[: int(num_prompts * random_selection_ensemble)]
            avg_probs = sum([all_avg_probs[jj] for jj in selected_prompts]) / len(selected_prompts)
            all_preds = [predictions[jj][-1] for jj in selected_prompts]
        else:
            avg_probs = avg_probs / num_prompts
            all_preds = [ppt[-1] for ppt in predictions]

        avg_label = np.argmax(avg_probs)
        counts = [all_preds.count(ii) + 1 for ii in range(num_targets)]
        vote_label = np.argmax(counts)
        total = float(sum(counts))
        vote_probs = [c / total for c in counts]

        if return_all_prompt_preds and num_examples == 1:
            if not self_train:
                random_indices = np.random.permutation(len(all_avg_probs))
            else:
                random_indices = np.arange(len(all_avg_probs))
            avg_probs = [all_avg_probs[ii] for ii in random_indices]
            vote_probs = [[1 if c == predictions[ii][-1] else 0 for c in range(num_targets)] for ii in random_indices]
            return [ppt[0] for ppt in predictions], avg_probs, vote_probs

        if pseudo_dist == "argmax":
            avg_probs = [1 if c == avg_label else 0 for c in range(num_targets)]
            vote_probs = [1 if c == vote_label else 0 for c in range(num_targets)]

        if num_examples == 1:
            avg_ensemble_predictions.append(avg_probs)
            vote_ensemble_predictions.append(vote_probs)
        else:
            avg_ensemble_predictions.append(avg_label)
            vote_ensemble_predictions.append(vote_label)

    if num_examples == 1:
        return [ppt[0] for ppt in predictions], avg_ensemble_predictions[0], vote_ensemble_predictions[0]

    prompt_metrics = []
    for ppred in predictions:
        m = metrics.compute(predictions=ppred, references=golds)
        p, r, f = _compute_precision_recall_f1(ppred, golds, num_targets)
        m.update({"precision": p, "recall": r, "f1": f})
        prompt_metrics.append(m)

    avg_ensemble_metrics = metrics.compute(predictions=avg_ensemble_predictions, references=golds)
    p, r, f = _compute_precision_recall_f1(avg_ensemble_predictions, golds, num_targets)
    avg_ensemble_metrics.update({"precision": p, "recall": r, "f1": f})
    avg_entropy = [np.mean(ents) for ents in entropies]
    vote_ensemble_metrics = metrics.compute(predictions=vote_ensemble_predictions, references=golds)
    p, r, f = _compute_precision_recall_f1(vote_ensemble_predictions, golds, num_targets)
    vote_ensemble_metrics.update({"precision": p, "recall": r, "f1": f})

    # print logits
    if fout_name.startswith("results"):
        nfout = fout_name + ".logits.p"
    else:
        nfout = os.path.join(fout_name, f"logits.{suffix}.p")
    for pidx in range(num_prompts):
        with open("{}{}".format(nfout, pidx), "w") as fout:
            for logit in logits[pidx]:
                fout.write(" ".join([str(l) for l in logit]) + "\n")

    posix_score = float(np.mean(posix_values))
    results = write_results_to_file(
        fout_name,
        suffix,
        prompt_metrics,
        predictions,
        avg_ensemble_metrics,
        avg_ensemble_predictions,
        vote_ensemble_metrics,
        vote_ensemble_predictions,
        golds,
        avg_entropy,
        extra_results={"posix": posix_score},
    )
    print(results)
    return results, None


def print_dict(dd):
    for key, value in dd.items():
        if isinstance(value, np.ndarray):
            print("{}: {}".format(key, " ".join([str(kk) for kk in value])))
        else:
            print("{}: {}".format(key, value))


def compute_entropy(predictions, num_targets):
    all_entropy = []
    for prompt_p in predictions:
        # import pdb; pdb.set_trace()
        prob = np.bincount(prompt_p, minlength=num_targets)
        prob = prob / len(prompt_p)
        all_entropy.append(scipy.stats.entropy(prob))

    return np.array(all_entropy)


def compute_unsupervised_metrics(
    logprobs,
    num_examples,
    num_targets,
    num_prompts,
    golds=None,
    metrics=None,
    fout_name=None,
    suffix=None,
    return_all_prompt_preds=False,
    random_selection_ensemble=0.0,
    initial_predictions=None,
    **kwargs
):

    # import pdb; pdb.set_trace()
    predictions = [[] for _ in range(num_prompts)]
    entropies = [[] for _ in range(num_prompts)]
    all_avg_probs = [[] for _ in range(num_prompts)]
    logits = [[] for _ in range(num_prompts)]
    posix_values = []
    idx = 0
    for eidx in range(num_examples):
        example_logits = []
        for pidx in range(num_prompts):
            max_ll, pred_label = -np.inf, -1
            # actually, the number of labels of each prompt should be the same
            normalized_probs = np.zeros(num_targets)
            logit = []
            for ii in range(num_targets):
                if logprobs[idx] > max_ll:
                    max_ll, pred_label = logprobs[idx], ii
                normalized_probs[ii] = math.exp(logprobs[idx])
                logit.append(logprobs[idx])
                idx += 1
            logits[pidx].append(logit)
            example_logits.append(logit)
            normalized_probs = normalized_probs / normalized_probs.sum()
            entropies[pidx].append(-(normalized_probs * np.log(normalized_probs)).sum())
            all_avg_probs[pidx].append(normalized_probs)
            predictions[pidx].append(pred_label)
        posix_values.append(_posix_from_logits(example_logits))

    results = {}

    entropy = compute_entropy(predictions, num_targets)
    results["all entropy"] = entropy
    results["avg entropy"] = entropy.mean()
    all_continuous_entropy = []
    for probs in all_avg_probs:
        all_continuous_entropy.append(scipy.stats.entropy(np.mean(probs, 0)))
    results["avg cont entropy"] = np.mean(all_continuous_entropy)

    fout_name = os.path.join(fout_name, f"unsupervised_dev_{suffix}")

    if golds is not None:
        avg_preds = []
        for eidx in range(num_examples):
            avg_p = np.mean([all_avg_probs[pidx][eidx] for pidx in range(num_prompts)], axis=0)
            avg_preds.append(int(np.argmax(avg_p)))
        m = metrics.compute(predictions=avg_preds, references=golds)
        p, r, f = _compute_precision_recall_f1(avg_preds, golds, num_targets)
        m.update({"precision": p, "recall": r, "f1": f})
        results.update(m)
        results["posix"] = float(np.mean(posix_values))

    if initial_predictions is None:
        print("finish collecting initial predictions before optimization")
        print_dict(results)
        write_unsupervised_results_to_file(fout_name, results, predictions, golds)
        return results, predictions
    else:
        initial_entropy = compute_entropy(initial_predictions, num_targets)
        results["delta all entropy"] = entropy - initial_entropy
        results["delta avg entropy"] = results["delta all entropy"].mean()
        print_dict(results)
        write_unsupervised_results_to_file(fout_name, results, predictions, golds)
        return results, None


def summarize_metrics(
    predictions, avg_ensemble_predictions, vote_ensemble_predictions, golds, metrics, fout_name=None
):
    prompt_metrics = []
    for ppred in predictions:
        prompt_metrics.append(metrics.compute(predictions=ppred, references=golds))
    avg_ensemble_metrics = metrics.compute(predictions=avg_ensemble_predictions, references=golds)
    vote_ensemble_metrics = metrics.compute(predictions=vote_ensemble_predictions, references=golds)

    results = {}
    for k, v in prompt_metrics[0].items():
        all_metrics = [pptm[k] * 100 for pptm in prompt_metrics]
        results["max_" + k] = round(np.max(all_metrics), 2)
        results["median_" + k] = round(np.median(all_metrics), 2)
        results["mean_" + k] = round(np.mean(all_metrics), 2)
        results["min_" + k] = round(np.min(all_metrics), 2)
        results["std_" + k] = round(np.std(all_metrics), 2)

    for k, v in avg_ensemble_metrics.items():
        results["avg_ensemble_avg" + k] = round(v * 100, 2)

    for k, v in vote_ensemble_metrics.items():
        results["vote_ensemble_avg" + k] = round(v * 100, 2)

    if fout_name is not None:
        _ = write_results_to_file(
            fout_name,
            "final",
            prompt_metrics,
            predictions,
            avg_ensemble_metrics,
            avg_ensemble_predictions,
            vote_ensemble_metrics,
            vote_ensemble_predictions,
            golds,
            extra_results=None,
        )
    return results


def compute_loss_scale(pred_labels, prompt_groups, group_id, answer_id):
    """
    compute how likely (unormalized) the prompts outside the current group supports the
    current answer
    """

    total = 0
    support = 0.0
    # for prompt_id, pred in enumerate(pred_labels):
    #     if prompt_id not in prompt_groups[group_id]:
    #         total += 1
    #         if pred == answer_id:
    #             support += 1.
    for prompt_id, pred in enumerate(pred_labels):
        total += 1
        if pred == answer_id:
            support += 1.0

    # only one group
    if total == 0:
        return 0

    return support


def compute_unsupervised_dev_best_results(dir_path, min_train_steps, metrics=["avg entropy", "avg cont entropy"]):
    unsup_dev_prefix = "unsupervised_dev_"
    eval_prefix = "accuracy_"
    all_checkpoints = []
    for fname in os.listdir(dir_path):
        if eval_prefix in fname:
            all_checkpoints.append(int(fname.split("_")[-1]))
    all_checkpoints.sort()

    best_ens_acc = 0.0
    best_ckpt = 0
    best_dev_results = {}
    all_results = {}
    for ckpt in all_checkpoints:
        with open(os.path.join(dir_path, eval_prefix + str(ckpt))) as fin:
            line = fin.readline()
            all_results[ckpt] = line.strip()
            line = line.strip().split(",")
            for field in line:
                k, v = field.split("=")
                v = float(v)
                if k == "avg_ensemble_accuracy" or k == "vote_ensemble_accuracy":
                    if v > best_ens_acc:
                        best_ckpt = ckpt
                        best_ens_acc = v

        if ckpt <= min_train_steps:
            continue
        if not os.path.exists(os.path.join(dir_path, unsup_dev_prefix + str(ckpt))):
            continue

        with open(os.path.join(dir_path, unsup_dev_prefix + str(ckpt))) as fin:
            for line in fin:
                if line.startswith("gold"):
                    break
                for metric in metrics:
                    if line.startswith(metric):
                        value = float(line.strip().split("=")[-1])
                        if metric in best_dev_results:
                            # larger metric is better: entropy
                            if value > best_dev_results[metric][-1]:
                                best_dev_results[metric] = (ckpt, value)
                        else:
                            best_dev_results[metric] = (ckpt, value)
    print("Best checkpoint at step {}: ".format(best_ckpt))
    print(all_results[best_ckpt])
    for k, v in best_dev_results.items():
        print("Best checkpoint selected by {} at step {}:".format(k, v[0]))
        print(all_results[v[0]])


def _compute_precision_recall_f1(preds: List[int], refs: List[int], num_labels: int):
    """Compute macro precision, recall and F1 scores."""
    tp = [0] * num_labels
    fp = [0] * num_labels
    fn = [0] * num_labels
    for p, r in zip(preds, refs):
        if p == r:
            tp[p] += 1
        else:
            fp[p] += 1
            fn[r] += 1
    precision = 0.0
    recall = 0.0
    for i in range(num_labels):
        denom_p = tp[i] + fp[i]
        denom_r = tp[i] + fn[i]
        precision += tp[i] / denom_p if denom_p > 0 else 0.0
        recall += tp[i] / denom_r if denom_r > 0 else 0.0
    precision /= num_labels
    recall /= num_labels
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def _posix_from_logits(prompt_logits: List[List[float]]) -> float:
    """Compute the POSIX metric from per-prompt logit lists."""
    N = len(prompt_logits)
    if N <= 1:
        if N == 0:
            raise ZeroDivisionError("No entries: cannot compute POSIX")
        logger.warning("Only one prompt provided. Returning 0.0 POSIX.")
        return 0.0

    best_classes = [int(np.argmax(row)) for row in prompt_logits]
    logprob_matrix = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            logprob_matrix[i][j] = prompt_logits[i][best_classes[j]]

    total = 0.0
    for j in range(N):
        diff_jj = logprob_matrix[j][j]
        for i in range(N):
            if i == j:
                continue
            diff_ij = logprob_matrix[i][j]
            total += abs(diff_ij - diff_jj)
    return total / (N * (N - 1))


def _fleiss_kappa_po(predictions: List[List[int]], num_labels: int) -> (float, float):
    """Compute Fleiss' kappa and raw agreement (P_o)."""
    import numpy as np

    N = len(predictions[0])  # number of items
    k = num_labels
    M = np.zeros((N, k), dtype=int)
    for i in range(N):
        for preds in predictions:
            M[i, preds[i]] += 1

    n_annotators = float(np.sum(M[0, :]))
    p = np.sum(M, axis=0) / (N * n_annotators)
    PbarE = np.sum(p * p)
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = np.sum(P) / N
    kappa = (Pbar - PbarE) / (1 - PbarE) if 1 - PbarE != 0 else 0.0
    return kappa, Pbar
