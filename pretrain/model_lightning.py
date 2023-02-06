from transformers import AdamW, BertForTokenClassification

import torch
import pytorch_lightning as pl
import logging


class LitBertTokenClassification(pl.LightningModule):
    def __init__(self, tokenizer, labels_dict, learning_rate, batch_size=1):
        super().__init__()
        print(f"initializing Bert... batch_size: {batch_size}, lr: {learning_rate}, num_labels: {len(labels_dict)}")
        self.labels_dict = labels_dict
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = BertForTokenClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=len(self.labels_dict))
        self.tokenizer = tokenizer
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self,
                input_ids,
                labels):
        assert len(input_ids) == len(labels)
        msg = "input shape :" + str(input_ids.shape)
        logging.info(msg)
        response = self.model(input_ids=input_ids,
                              labels=labels)
        return response

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch["input_ids"], batch["labels"]
        msg = "input shape :" + str(input_ids.shape) + "labels batch: " + str(labels.shape)
        logging.info(msg)
        output = self.model(input_ids=input_ids,
                            labels=labels)
        loss = output.loss
        # Logging to Tensorboard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch["input_ids"], batch["labels"]
        output = self.model(input_ids=input_ids,
                            labels=labels)
        val_loss = output.loss
        val_logits = output.logits
        self.log("val_loss", val_loss)
        return {"loss": val_loss, "logits": val_logits, "label": labels}

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        input_ids, labels = batch["input_ids"], batch["labels"]
        output = self.model(input_ids=input_ids)
        new_dict = dict((v, k) for k, v in self.labels_dict.items())
        # print(new_dict)
        results = []
        for i in range(len(input_ids)):
            token_num = len(input_ids[i])
            # reconstruct text and labels without bos and eos
            input_token = self.tokenizer.convert_ids_to_tokens(input_ids[i])[1:token_num-1]
            gt = [new_dict[label_id] for label_id in labels[i][1:token_num-1].tolist()]
            # decode output
            m = torch.nn.Softmax(dim=1)
            label_ids = m(output.logits[i]).tolist()
            prediction = [new_dict[label_id.index(max(label_id))] for label_id in label_ids][1: token_num-1]
            assert len(gt) == len(prediction)
            result = {
                "text": ' '.join(input_token),
                "gt": gt,
                "prediction": prediction}
            # print("-----result----\n")
            results.append(result)
        return results

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=True)
        return optimizer
