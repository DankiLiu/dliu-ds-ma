from typing import List

import torch
from transformers import BertModel


class MultiTaskBert(torch.nn.Module):
    def __init__(self, tokenizer, encoder_name, tasks: List, classifier_only):
        super().__init__()
        self.encoder = BertModel.from_pretrained(encoder_name)

        self.output_heads = torch.nn.ModuleDict()
        for task in tasks:
            decoder = self._create_output_head(self.encoder.config.hidden_size, task)
            self.output_heads[str(task.id)] = decoder

        self.tokenizer = tokenizer
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        if classifier_only:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def set_output_heads(self, tasks):
        for task in tasks:
            decoder = self._create_output_head(self.encoder.config.hidden_size, task)
            self.output_heads[str(task.id)] = decoder

    @staticmethod
    def _create_output_head(encoder_hidden_size: int, task):
        if task.type == "seq_classification":
            return SequenceClassificationHead(encoder_hidden_size, task.num_labels)
        elif task.type == "tok_classification":
            return TokenClassificationHead(encoder_hidden_size, task.num_labels)
        else:
            raise NotImplementedError

    def forward(self, input_ids, labels, task_ids, attention_mask=None):
        outputs = self.encoder(input_ids)
        sequence_output, pooled_output = outputs[:2]

        loss_list = []
        logits = None
        for task_id in task_ids:
            logits, task_loss = self.output_heads[str(task_id)].forward(
                sequence_output,
                pooled_output,
                labels=None if labels is None else labels[task_id],
                attention_mask=None if attention_mask is None else attention_mask[task_id],
            )

            if labels is not None:
                loss_list.append(task_loss)

        outputs = (logits, outputs[2:])

        if loss_list:
            loss = torch.stack(loss_list)
            outputs = (loss.mean(),) + outputs

        return outputs

    def prediction(self, input_ids, labels, task_ids):
        outputs = self.encoder(input_ids)
        sequence_output, pooled_output = outputs[:2]

        logits_dict = {}
        for task_id in task_ids:
            logits, _ = self.output_heads[str(task_id)].forward(
                sequence_output,
                pooled_output,
                labels=None if labels is None else labels[task_id]
            )

            # if labels is not None:
            logits_dict[task_id] = logits

        return logits_dict


class TokenClassificationHead(torch.nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_p=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
        self.classifier = torch.nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self._init_weights()

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(
            self, sequence_output, pooled_output, labels=None, attention_mask=None, **kwargs
    ):
        sequence_output_dropout = self.dropout(sequence_output)
        logits = self.classifier(sequence_output_dropout)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()

            labels = labels.long()

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return logits, loss


class SequenceClassificationHead(torch.nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_p=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.dropout = torch.nn.Dropout(dropout_p)
        self.hidden_size = hidden_size
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

        self._init_weights()

    def forward(self, sequence_output, pooled_output, labels=None, **kwargs):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if labels.dim() != 1:
                # Remove padding
                labels = labels[:, 0]

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.long().view(-1)
            )

        return logits, loss

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()


class Task:
    def __init__(self, task_id, task_name, labels_dict, type):
        self.id = task_id
        self.name = task_name
        self.labels_dict = labels_dict
        self.num_labels = len(labels_dict)
        self.type = type

    def print(self):
        print(f"task_id: {self.id}")
        print(f"task_name: {self.name}")
        print(f"num of labels: {self.num_labels}")
        print(f"task_type: {self.type}")
