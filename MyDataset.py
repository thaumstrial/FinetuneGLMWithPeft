import random
from dataclasses import dataclass
from typing import Optional, List, Dict
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, BatchEncoding


class CLMDataset(Dataset):
    def __init__(self):
        self.queries = []
        self.responses = []

        data = [
            ([], '你是谁？\n [gMASK] ', '我也不知道我是谁。')
        ]

        for history, query, response in data:

            if not history:
                prompt = query
            else:
                prompt = ""
                for i, (old_query, response) in enumerate(history):
                    prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
                prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

            self.queries.append(prompt)
            self.responses.append(response)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        return self.queries[index], self.responses[index]

@dataclass
class CLMDataCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None

    def __call__(self, batch):
        input_ids = []
        labels = []
        for query, response in batch:
            input_ids.append(query)
            labels.append(response)

        input_ids = self.tokenizer(input_ids, return_tensors="pt",
                                   padding='max_length', max_length=self.max_length, truncation=True).input_ids
        labels = self.tokenizer(labels, return_tensors="pt",
                                   padding='max_length', max_length=self.max_length, truncation=True).input_ids
        return {"input_ids": input_ids, "labels": labels}
