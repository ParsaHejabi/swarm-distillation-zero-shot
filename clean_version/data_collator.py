from typing import List
from transformers import DataCollatorForSeq2Seq

class ListDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    """Extension of HuggingFace's DataCollatorForSeq2Seq that handles nested lists."""
    def __init__(self, *args, expand_list: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.expand_list = expand_list

    def __call__(self, features, return_tensors=None):
        if self.expand_list and features and isinstance(features[0], list):
            assert len(features) == 1, "Batch size >1 not supported for nested lists"
            collated_batches = []
            for sublist in features[0]:
                if isinstance(sublist, list):
                    collated_batches.append(super().__call__(sublist, return_tensors))
                else:
                    collated_batches.append(super().__call__([sublist], return_tensors))
            return collated_batches
        return super().__call__(features, return_tensors)