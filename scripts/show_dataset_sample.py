"""Inspect how :class:`DatasetByPrompt` transforms raw dataset examples."""

import sys
from pathlib import Path

import datasets
from transformers import AutoTokenizer, HfArgumentParser
from promptsource.templates import DatasetTemplates

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ttt.dataloader import DatasetByPrompt
from ttt.options import DataArguments


def main():
    parser = HfArgumentParser(DataArguments)
    if len(sys.argv) == 1:
        parser.print_help()
        return

    data_args = parser.parse_args_into_dataclasses()[0]

    subset = data_args.subset_name if data_args.subset_name != "none" else None

    # load the raw dataset split
    raw_dataset = datasets.load_dataset(
        data_args.dataset_name,
        subset,
        split=data_args.testset_name,
    )
    example = raw_dataset[0]
    print("Raw example:\n", example, "\n")

    # show how each template formats the same example
    templates = DatasetTemplates(data_args.prompt_set_name, subset)
    prompt_names = [
        name for name in templates.all_template_names
        if templates[name].metadata.original_task
    ]

    print(f"Applying {len(prompt_names)} templates:\n")
    for name in prompt_names:
        input_text, output_text = templates[name].apply(example)
        choices = templates[name].get_answer_choices_list(example)
        print(f"Template: {name}")
        print("Input:", input_text)
        print("Output:", output_text)
        if choices is not None:
            print("Choices:", choices)
        print("-" * 40)

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    proc_dataset = DatasetByPrompt(
        data_args,
        cache_dir="./cache",
        tokenizer=tokenizer,
        split=data_args.testset_name,
    )

    examples, label = proc_dataset[0]
    print("\nAfter DatasetByPrompt -- tokenized inputs (showing first):")
    print(examples[0])
    print("Number of prompt variations:", len(examples))
    print("Gold label:", label)


if __name__ == "__main__":
    main()
