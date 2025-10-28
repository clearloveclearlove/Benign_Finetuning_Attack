import json
import torch
from torch.utils.data import Dataset
from typing import List
from sentencepiece import SentencePieceProcessor

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

PROMPT_DICT = {
    "prompt_context": (
        B_SYS + "Below is an instruction that describes a task. " +
        "Write a response that appropriately completes the request." + E_SYS +
        "### Instruction:\n{instruction}\n\nInput:\n{context}\n\n### Response:\n"
    ),
    "prompt_no_context": (
        B_SYS + "Below is an instruction that describes a task. " +
        "Write a response that appropriately completes the request." + E_SYS +
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}

class GSM8KDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=30, pad=True):
        file_path = dataset_config.data_path
        self.ann = self._load_data(file_path)

        if partition == "train":
            self.ann = self.ann[0:]
        else:
            self.ann = self.ann[-200:]

        self.max_words = max_words
        self.tokenizer = tokenizer
        self.pad = pad

    def _load_data(self, file_path):
        with open(file_path, "r") as file:
            data = file.read().strip().split("\n")
        return [json.loads(entry) for entry in data]

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        question = ann["question"]
        answer = ann["answer"]

        prompt = B_INST + " " + PROMPT_DICT["prompt_no_context"].format(instruction=question) + " " + E_INST
        example = prompt + " " + answer + " "

        input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        example_ids = self.tokenizer.encode(example)
        example_ids.append(self.tokenizer.eos_token_id)
        example_tensor = torch.tensor(example_ids, dtype=torch.int64)

        if self.pad:
            padding = self.max_words - example_tensor.shape[0]
            if padding > 0:
                example_tensor = torch.cat((example_tensor, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                example_tensor = example_tensor[:self.max_words]

        labels = example_tensor.clone()
        labels[:len(input_ids)] = -1

        example_mask = example_tensor.ge(0).float()
        label_mask = labels.ge(0).float()

        example_tensor[~example_mask.bool()] = 0
        labels[~label_mask.bool()] = -100

        return {
            "input_ids": example_tensor,
            "labels": labels,
            "attention_mask": example_mask,
        }

def get_gsm8k_dataset(dataset_config, tokenizer, partition="train", max_words=30, concat=False):
    return GSM8KDataset(dataset_config, tokenizer, partition, max_words, pad=True)
