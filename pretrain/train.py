# import sys
# adding Folder_2/subfolder to the system path
# sys.path.insert(0, '/home/daliu/Documents/master-thesis/code/dliu-ds-ma')

from pytorch_lightning import Trainer, tuner
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer, BertTokenizer

from data.data_processing import store_jointslu_labels
from pretrain.model_lightning import LitBertTokenClassification
from pretrain.jointslu_data_module import JointsluDataModule
from pathlib import Path


# Update labels again if training dataset is generated
def update_label():
    store_jointslu_labels()


def define_tokenizer():
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        do_lower_case=True,
        sep_token='EOS',
        cls_token='BOS')
    # Setup tokenizer
    return tokenizer


def train():
    tokenizer = define_tokenizer()
    data_module = JointsluDataModule(tokenizer=tokenizer)
    model = LitBertTokenClassification(tokenizer=tokenizer)

    logger = TensorBoardLogger("pretrain/model_sim", name="bert_jointslu")
    trainer = Trainer(max_epochs=3, logger=logger)
    trainer.fit(model, datamodule=data_module)


def auto_lr_bz_train():
    """train model with best batch size and learning rate found by lightning"""
    tokenizer = define_tokenizer()
    model = LitBertTokenClassification(tokenizer=tokenizer)
    data_module = JointsluDataModule(tokenizer=tokenizer)

    logger = TensorBoardLogger("pretrain/auto_sim", name="bert_jointslu")
    trainer = Trainer(auto_lr_find=True,
                      auto_scale_batch_size=True,
                      max_epochs=3,
                      logger=logger)
    trainer.tune(model, datamodule=data_module)
    print(f"lr {model.learning_rate} bz {model.batch_size}")
    trainer.fit(model, datamodule=data_module)


def load_from_checkpoint():
    tokenizer = define_tokenizer()
    model = LitBertTokenClassification(tokenizer=tokenizer)
    data_module = JointsluDataModule(tokenizer=tokenizer)
    trainer = Trainer()
    current_path =str(Path().absolute())
    chk_path = Path(current_path + "/pretrain/model_sim/bert_jointslu/version_0/checkpoints/epoch=2-step=11946.ckpt")
    print(chk_path)
    trained_model = model.load_from_checkpoint(chk_path, tokenizer=tokenizer)
    results = trainer.test(model=trained_model, datamodule=data_module, verbose=True)

    # prediction =
    print(results)


if __name__ == '__main__':
    auto_lr_bz_train()
