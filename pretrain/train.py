import sys

# adding Folder_2/subfolder to the system path
sys.path.insert(0, '/home/daliu/Documents/master-thesis/code/dliu-ds-ma')

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import GPT2Tokenizer

from data.data_processing import store_jointslu_labels
from pretrain.model_lightning import LitGpt2Prediction
from pretrain.jointslu_data_module import JointsluDataModule
from util import read_jointslu_labels


# Update labels again if training dataset is generated
def update_label():
    store_jointslu_labels()


def define_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    special_tokens = read_jointslu_labels()
    # Setup tokenizer
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.add_special_tokens({'bos_token': 'BOS',
                                  'eos_token': 'EOS',
                                  'pad_token': 'PAD'})
    print("Created tokenizer with special tokens")
    return tokenizer


if __name__ == '__main__':
    update_label()
    tokenizer = define_tokenizer()
    data_module = JointsluDataModule(tokenizer=tokenizer)
    model = LitGpt2Prediction(tokenizer=tokenizer)

    logger = TensorBoardLogger("pretrain/tb_logger", name="gpt2_jointslu")
    trainer = Trainer(max_epochs=3, logger=logger)
    trainer.fit(model, datamodule=data_module)
