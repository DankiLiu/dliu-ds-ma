import json
from typing import Dict, Union, List
from typing import Literal

import torch
from tokenizers import Tokenizer
from torch import Tensor
from torch.utils.data import Dataset
from transformers import BatchEncoding
from transformers.models.hubert.modeling_tf_hubert import input_values_processing

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
        input_tok_b = [text.split(' ') for text in texts_b]
        labels_tok_b = [label.split(' ') for label in labels_b]
        input_ids = [self.tokenizer.convert_tokens_to_ids(input_tok)
                     for input_tok in input_tok_b]
        labels_ids = [self.tokenizer.convert_tokens_to_ids(labels_tok)
                      for labels_tok in labels_tok_b]
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
            path = '../data/jointslu/train.json'
        elif split == 'test':
            path = '../data/jointslu/test.json'
        elif split == 'val':
            path = '../data/jointslu/val.json'
        if path:
            with open(path) as f:
                annotations = json.load(f)
                return cls(tokenizer, split, annotations)
        else:
            print(f"{split} path does not exist")
            return None
