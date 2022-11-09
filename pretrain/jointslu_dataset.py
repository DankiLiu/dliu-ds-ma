import json
from typing import Dict, Union, List
from typing import Literal

import torch
from tokenizers import Tokenizer
from torch import Tensor
from torch.utils.data import Dataset
from transformers import BatchEncoding

DiaAnnotationFileDictKey = Literal[
    'id',
    'text',
    'labels'
]

DiaSampleDictKey = Literal[
    'text',
    'labels'
]

DiaBatchDictKey = Literal[
    'batch_text',  # b * L
    'batch_labels'  # B * L
]
if False:
    # Create labels dictionary
    f = open("../data/jointslu/labels.csv")
    import pandas as pd
    labels_list = pd.read_csv(f, usecols=["simplified label"]).values
    labels = list(set(item for sublist in labels_list for item in sublist))
    labels_dict = {labels[i]: i for i in range(len(labels))}
    # save into a file
    with open("../data/jointslu/pre-train/labels.json", 'w') as f:
        json.dump(labels_dict, f, indent=4)
# Load labels
f = open("../data/jointslu/pre-train/labels.json", 'r')
labels_dict = json.load(f)
print(f"label size is {len(labels_dict)}")
f.close()
DatasetSplitName = Literal["train", "val", "test"]
DiaSample = Dict[str, Union[torch.Tensor, str, int]]
DiaBatch = Dict[str, Union[List, Tensor]]


class JointSluDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 split,
                 annotations: List[Dict[str, Union[str, int]]]):
        self.tokenizer = tokenizer
        self.split = split
        self.annotations = annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index) -> DiaSample:
        text = self.annotations[index]['text']
        labels = self.annotations[index]['labels']

        sample: DiaSample = {
            'text': text,
            'labels': labels
        }
        return sample

    def collate_dia_samples(self, batch: List[DiaSample]) -> DiaBatch:
        # b['text'] and b['labels'] are two lists
        input_tok_b = [b['text'] for b in batch]
        labels_tok_b = [b['labels'] for b in batch]
        # print(f"input_tok {input_tok_b}")
        # print(F"labels_tok {labels_tok_b}")
        input_ids = [self.tokenizer.convert_tokens_to_ids(input_tok)
                     for input_tok in input_tok_b]

        labels_ids = []
        for labels_tok in labels_tok_b:
            label_ids = []
            tok_length = len(labels_tok)
            for i in range(tok_length):
                # first and last token is CLS and SEP, set as -100
                if i == 0:
                    label_ids.append(-100)
                elif i == tok_length - 1:
                    label_ids.append(-100)
                else:
                    try:
                        label_ids.append(labels_dict[labels_tok[i]])
                    except KeyError:
                        print("Label not found")
                        label_ids.append(len(labels_dict))
            labels_ids.append(label_ids)

        batch = BatchEncoding({"input_ids": input_ids,
                               "labels": labels_ids})
        tensor_batch = batch.convert_to_tensors(tensor_type="pt")
        result: DiaBatch = tensor_batch
        return result

    @classmethod
    def create_data(cls, split: DatasetSplitName,
                    tokenizer: Tokenizer):
        path = None
        if split == 'train':
            path = '../data/jointslu/pre-train/b_train.json'
        elif split == 'test':
            path = '../data/jointslu/pre-train/b_test.json'
        elif split == 'val':
            path = '../data/jointslu/pre-train/b_val.json'
        if path:
            with open(path) as f:
                annotations = json.load(f)
                return cls(tokenizer, split, annotations)
        else:
            print(f"{split} path does not exist")
            return None
