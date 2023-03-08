import json
from typing import Dict, Union, List
from typing import Literal

import torch
from tokenizers import Tokenizer
from torch import Tensor
from torch.utils.data import Dataset
from transformers import BatchEncoding

from data.data_processing import get_labels_dict, get_intents_dict

DiaAnnotationFileDictKey = Literal[
    'id',
    'text',
    'intent',
    'labels'
]

DiaSampleDictKey = Literal[
    'text',
    'intent',
    'labels'
]

DiaBatchDictKey = Literal[
    'batch_text',  # b * L
    'batch_intents',  # B * L
    'batch_labels'
]
DatasetSplitName = Literal["train", "val", "test"]
DiaSample = Dict[str, Union[torch.Tensor, str, int]]
DiaBatch = Dict[str, Union[List, Tensor]]


class MTDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 split,
                 annotations: List[Dict[str, Union[str, int]]],
                 intents_dict,
                 labels_dict):
        self.tokenizer = tokenizer
        self.split = split
        self.annotations = annotations
        self.intents_dict = intents_dict
        self.labels_dict = labels_dict

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index) -> DiaSample:
        text = self.annotations[index]['text']
        intent = self.annotations[index]['intent']
        labels = self.annotations[index]['labels']

        sample: DiaSample = {
            'text': text,
            'intent': intent,
            'labels': labels
        }
        return sample

    def collate_dia_samples(self, batch: List[DiaSample]) -> DiaBatch:
        # b['text'] and b['labels'] are two lists
        input_tok_b = [b['text'] for b in batch]
        intent_tok_b = [b['intent'] for b in batch]
        labels_tok_b = [b['labels'] for b in batch]

        input_ids = [self.tokenizer.convert_tokens_to_ids(input_tok)
                     for input_tok in input_tok_b]
        # pad the intent to the same length as input
        # the first element is integer for intent label, then pad with -100
        intent_ids = []
        for i, labels_tok in enumerate(intent_tok_b):
            padding_length = len(input_ids[i]) - 1
            padding = [-100] * padding_length
            intent_id = len(self.intents_dict)
            try:
                intent_id = self.intents_dict[labels_tok]
            except KeyError:
                print(f"{intent_id} does not exist, intent label is {len(self.intents_dict)}")
            padding.insert(0, intent_id)
            intent_ids.append(padding)

        # replace the first and last element of labels with -100 and replace other with integers
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
                               "intents": intent_ids,
                               "labels": labels_ids})
        tensor_batch = batch.convert_to_tensors(tensor_type="pt")
        result: DiaBatch = tensor_batch
        return result

    @classmethod
    def create_data(cls, dataset, labels_version, scenario, split: DatasetSplitName,
                    tokenizer: Tokenizer, few_shot_num):
        # load labels_dict
        intents_dict = get_intents_dict(dataset=dataset, labels_version=labels_version, scenario=scenario)
        labels_dict = get_labels_dict(dataset=dataset, labels_version=labels_version, scenario=scenario)
        data_folder = "data/" + dataset + "/training_data/labels" + labels_version + '/'

        path = None
        if split == 'train':
            if few_shot_num != -1 and few_shot_num is not None:
                path = data_folder + few_shot_num + "train.json" if not dataset == "massive" \
                    else data_folder + scenario + str(few_shot_num) + "train.json"
            else:
                path = data_folder + "train.json" if not dataset == "massive" \
                    else data_folder + scenario + "_train.json"
        elif split == 'test':
            path = data_folder + "test.json" if not dataset == "massive" else data_folder + scenario + "_test.json"
        elif split == 'val':
            path = data_folder + "val.json" if not dataset == "massive" else data_folder + scenario + "_val.json"
        if path:
            with open(path, 'r') as f:
                annotations = json.load(f)
                return cls(tokenizer, split, annotations,
                           labels_dict=labels_dict,
                           intents_dict=intents_dict)
        else:
            print(f"{split} path does not exist")
            return None
