"""Inspect how :class:`DatasetByPrompt` transforms raw dataset examples."""

import sys
from pathlib import Path

import datasets
from transformers import AutoTokenizer, HfArgumentParser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ttt.dataloader import DatasetByPrompt
from ttt.options import DataArguments


def main():
    parser = HfArgumentParser(DataArguments)
    if len(sys.argv) == 1:
        parser.print_help()
        return

    data_args = parser.parse_args_into_dataclasses()[0]

    # load the raw dataset split
    raw_dataset = datasets.load_dataset(
        data_args.dataset_name,
        data_args.subset_name if data_args.subset_name != "none" else None,
        split=data_args.testset_name,
    )
    print("Raw example:\n", raw_dataset[0], "\n")

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    proc_dataset = DatasetByPrompt(
        data_args,
        cache_dir="./cache",
        tokenizer=tokenizer,
        split=data_args.testset_name,
    )
    
    examples, label = proc_dataset[0]
    print("After DatasetByPrompt (first prompt example):\n", examples[0])
    print("Gold label:", label)


if __name__ == "__main__":
    main()
