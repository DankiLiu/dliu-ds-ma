from transformers import AdamW

import torch
import pytorch_lightning as pl


from pretrain.multi_task.multi_task_bert import MultiTaskBert


class LitBertMultiTask(pl.LightningModule):
    def __init__(self, tokenizer, tasks, classifier_only, learning_rate=2.2908676527677725e-05, batch_size=1):
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tasks = tasks
        self.tokenizer = tokenizer
        self.classifier_only = classifier_only
        self.multi_task_bert = MultiTaskBert(tokenizer, 'bert-base-uncased', tasks)

    def forward(self, input_ids, task_ids, attention_mask):
        # todo: one task or two at the same time?
        outputs = self.multi_task_bert.forward(input_ids, None, task_ids, attention_mask)
        return {'logits': outputs}

    def training_step(self, batch, batch_idx):
        input_ids, intents, labels = batch['input_ids'], batch['intents'], batch['labels']

        annotations = []
        task_ids = []
        for task in self.tasks:
            task_ids.append(task.id)
            if task.type == "tok_classification":
                annotations.append(labels)
            if task.type == "seq_classification":
                annotations.append(intents)

        loss, logits, outputs = self.multi_task_bert.forward(input_ids, annotations, task_ids)

        self.log("train loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, intents, labels = batch["input_ids"], batch["intents"], batch["labels"]
        task_ids = range(len(self.tasks))
        loss, logits, outputs = self.multi_task_bert.forward(input_ids=input_ids,
                                                             labels=[intents, labels],
                                                             task_ids=task_ids)
        self.log("val_loss", loss)
        return {"loss": loss, "logits": logits, "output": outputs}

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        input_ids, intents, labels = batch['input_ids'], batch['intents'], batch['labels']
        annotations = []
        task_ids = []
        for task in self.tasks:
            task_ids.append(task.id)
            if task.type == "tok_classification":
                annotations.append(labels)
            if task.type == "seq_classification":
                annotations.append(intents)
        logits_dict = self.multi_task_bert.prediction(input_ids, None, task_ids)
        tok_predictions, seq_predictions = [], []
        input_tokens, intent_gts, labels_gts = [], [], []
        for task in self.tasks:
            logits = logits_dict[task.id]
            new_dict = dict((v, k) for k, v in task.labels_dict.items())
            input_tokens = [self.tokenizer.convert_ids_to_tokens(ids)[1:-1] for ids in input_ids]
            if task.type == "tok_classification":
                for label in labels:
                    for label_id in label[1:-1].tolist():
                        if label_id in new_dict.keys():
                            labels_gts.append(new_dict[label_id])
                        else:
                            labels_gts.append('O')
                self.decode_tok_classification(new_dict, input_ids, logits, tok_predictions)
            elif task.type == "seq_classification":
                intent_gts = []
                for intent in intents:
                    if int(intent[0]) in new_dict.keys():
                        intent_gts.append(new_dict[int(intent[0])])
                    else:
                        intent_gts.append("unknown")
                self.decode_seq_classification(new_dict, input_ids, logits, seq_predictions)

        assert len(input_ids) == len(tok_predictions) == len(seq_predictions)
        result = {
            "input_tokens": input_tokens,
            "intent_gts": intent_gts,
            "labels_gts": labels_gts,
            "tok_predictions": tok_predictions,
            "seq_predictions": seq_predictions
        }
        return result

    def decode_tok_classification(self, new_dict, input_ids, logits, tok_predictions):
        # decode according to task type
        for i in range(len(input_ids)):
            # decode output
            m = torch.nn.Softmax(dim=1)
            label_ids = m(logits[i]).tolist()
            prediction = []
            for label_id in label_ids[1: -1]:
                if label_id.index(max(label_id)) in new_dict.keys():
                    prediction.append(new_dict[label_id.index(max(label_id))])
                else:
                    prediction.append("unknown")
            tok_predictions.append(prediction)

    def decode_seq_classification(self, new_dict, input_ids, logits, seq_predictions):
        for i in range(len(input_ids)):
            # reconstruct text and labels without bos and eos
            m = torch.nn.Softmax(dim=0)
            label_ids = m(logits[i]).tolist()
            prediction = "unknown"
            if label_ids.index(max(label_ids)) in new_dict.keys():
                prediction = new_dict[label_ids.index(max(label_ids))]
            seq_predictions.append(prediction)

    def configure_optimizers(self):
        params = []
        if self.classifier_only:
            for name, param in self.named_parameters():
                if 'multi_task_bert.output_heads' in name:
                    params.append(param)
                else:
                    param.requires_grad = False
        else:
            for name, param in self.named_parameters():
                params.append(param)
        optimizer = AdamW(params, lr=self.learning_rate, correct_bias=True)
        return optimizer
