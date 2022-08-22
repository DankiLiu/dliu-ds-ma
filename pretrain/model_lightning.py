from transformers import AdamW, GPT2LMHeadModel, GPT2Tokenizer
import pytorch_lightning as pl


class LitGpt2Prediction(pl.LightningModule):
    def __init__(self, tokenizer, learning_rate=2.2908676527677725e-05, batch_size=1):
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = tokenizer
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self,
                input_ids,
                labels):
        print(f"input_ids: {input_ids}")
        print(f"labels   : {labels}")
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
        print(input_ids)
        print(labels)
        output = self.model(input_ids=input_ids,
                            labels=labels)
        val_loss = output.loss
        val_logits = output.logits
        self.log("val_loss", val_loss)
        return {"loss": val_loss, "logits": val_logits, "label": labels}

    def test_step(self, batch, batch_idx):
        input_ids, labels = batch["input_ids"], batch["labels"]
        output = self.model.generate(input_ids=input_ids)

        labels = self.tokenizer.convert_ids_to_tokens(labels[0])
        # self.log("output", res_sen)
        input_text = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        output_labels = self.tokenizer.convert_ids_to_tokens(labels[0])
        output = {"input_text": input_text,
                  "labels": output_labels}
        print(output["input_text"])
        print(output["labels"])
        return output

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=True)
        return optimizer
