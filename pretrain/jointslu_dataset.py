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
f = open("../data/jointslu_dict.json")
labels = json.load(f)
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
        texts_b = [b['text'] for b in batch]
        labels_b = [b['labels'] for b in batch]
        input_tok_b = [text.split(' ') for text in texts_b]
        labels_tok_b = [label.split(' ') for label in labels_b]
        print(f"input_tok {input_tok_b}")
        print(F"labels_tok {labels_tok_b}")
        input_ids = [self.tokenizer.convert_tokens_to_ids(input_tok)
                     for input_tok in input_tok_b]
        #labels_ids = [self.tokenizer.convert_tokens_to_ids(labels_tok)
        #              for labels_tok in labels_tok_b]
        labels_ids = []
        for labels_tok in labels_tok_b:
            label_ids = []
            tok_length = len(labels_tok)
            for i in range(tok_length):
                if i == 0:
                    label_ids.append(-100)
                elif i == tok_length - 1:
                    label_ids.append(-100)
                else:
                    try:
                        label_ids.append(labels[labels_tok[i]])
                    except:
                        print("Label not found")
                        label_ids.append(138)
            labels_ids.append(label_ids)
            print("encoded labels ids is: ", labels_ids)
        print(labels_tok_b)

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
            path = '../data/jointslu/bert_train.json'
        elif split == 'test':
            path = '../data/jointslu/bert_test.json'
        elif split == 'val':
            path = '../data/jointslu/bert_val.json'
        if path:
            with open(path) as f:
                annotations = json.load(f)
                return cls(tokenizer, split, annotations)
        else:
            print(f"{split} path does not exist")
            return None
