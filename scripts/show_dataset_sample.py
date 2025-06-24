import datasets
from argparse import Namespace
from transformers import AutoTokenizer
from ttt.dataloader import DatasetByPrompt


def main():
    # load the raw validation example from SuperGLUE RTE
    raw_dataset = datasets.load_dataset("super_glue", "rte", split="validation")
    print("Raw example:\n", raw_dataset[0], "\n")

    # build DatasetByPrompt using the same split
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    args = Namespace(
        dataset_name="super_glue",
        subset_name="rte",
        testset_name="validation",
        prompt_set_name="super_glue",
        task_type="classification",
        cb_surgery=0,
        abl_nprompts=-1,
    )
    proc_dataset = DatasetByPrompt(args, cache_dir="./cache", tokenizer=tokenizer, split="validation")

    examples, label = proc_dataset[0]
    print("After DatasetByPrompt (first prompt example):\n", examples[0])
    print("Gold label:", label)


if __name__ == "__main__":
    main()
