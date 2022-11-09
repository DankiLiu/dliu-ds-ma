from transformers import AdamW, BertForTokenClassification

import torch
import pytorch_lightning as pl
from data.data_processing import read_jointslu_labels_dict as lbd


class LitBertTokenClassification(pl.LightningModule):
    def __init__(self, tokenizer, learning_rate=2.2908676527677725e-05, batch_size=1):
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = BertForTokenClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=69)
        self.tokenizer = tokenizer
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self,
                input_ids,
                labels):
        print(f"input_ids: {input_ids}")
        print(f"labels   : {labels}")
        assert len(input_ids) == len(labels)
        response = self.model(input_ids=input_ids,
                              labels=labels)
        return response

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch["input_ids"], batch["labels"]
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
        input_ids, labels = batch["input_ids"], batch["labels"]
        output = self.model(input_ids=input_ids)
        print(input_ids)
        print(output)
        # decode the labels
        labels_dict = lbd()
        new_dict = dict((v, k) for k, v in labels_dict.items())
        results = []
        for i in range(len(input_ids)):
            # for each example
            m = torch.nn.Softmax(dim=1)
            label_ids = m(output.logits[i]).tolist()
            print(label_ids)
            print(f"label ids after softmax \nlabel ids: {label_ids}")
            predictions = [labels_dict.index(label_id) for label_id in label_ids]
            print("predictions shape ", len(predictions))
            print(predictions)
            exit()
            labels = []
            input_token = self.tokenizer.convert_ids_to_tokens(input_ids[i])
            for index in predictions:
                labels.append(new_dict[index])

            result = {"input_token": input_token,
                      "labels": labels}
            print("-----result----\n")
            print(result)
            results.append(result)
        return results

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=True)
        return optimizer
