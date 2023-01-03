import json
from typing import Dict, Union, List
from typing import Literal

import torch
from tokenizers import Tokenizer
from torch import Tensor
from torch.utils.data import Dataset
from transformers import BatchEncoding

from data.data_processing import get_labels_dict

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
DatasetSplitName = Literal["train", "val", "test"]
DiaSample = Dict[str, Union[torch.Tensor, str, int]]
DiaBatch = Dict[str, Union[List, Tensor]]


class JointSluDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 split,
                 annotations: List[Dict[str, Union[str, int]]],
                 labels_dict):
        self.tokenizer = tokenizer
        self.split = split
        self.annotations = annotations
        self.labels_dict = labels_dict

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
                        label_ids.append(self.labels_dict[labels_tok[i]])
                    except KeyError:
                        print("Label not found")
                        label_ids.append(len(self.labels_dict))
            labels_ids.append(label_ids)

        batch = BatchEncoding({"input_ids": input_ids,
                               "labels": labels_ids})
        tensor_batch = batch.convert_to_tensors(tensor_type="pt")
        result: DiaBatch = tensor_batch
        return result

    @classmethod
    def create_data(cls, dataset, labels_version, split: DatasetSplitName,
                    tokenizer: Tokenizer):
        # load labels_dict
        labels_dict = get_labels_dict(dataset=dataset, labels_version=labels_version)
        data_folder = "data/" + dataset + "/training_data/labels" + labels_version
        path = None
        # todo: if path not exist, construct training data. or need I?
        if split == 'train':
            path = data_folder + "/train.json"
        elif split == 'test':
            path = data_folder + "/test.json"
        elif split == 'val':
            path = data_folder + "/val.json"
        if path:
            with open(path, 'r') as f:
                annotations = json.load(f)
                return cls(tokenizer, split, annotations, labels_dict)
        else:
            print(f"{split} path does not exist")
            return None
