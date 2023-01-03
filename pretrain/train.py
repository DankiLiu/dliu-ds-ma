# import sys
# adding Folder_2/subfolder to the system path
# sys.path.insert(0, '/home/daliu/Documents/master-thesis/code/dliu-ds-ma')

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import BertTokenizer
from data.data_processing import store_jointslu_labels, get_labels_dict
from pretrain.model_lightning import LitBertTokenClassification
from pretrain.jointslu_data_module import JointsluDataModule
from evaluation.evaluation_utils import get_std_gt
from pathlib import Path


# Update labels again if training dataset is generated
from util import append_to_json


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


def train(model_version, dataset, labels_version):
    labels_dict = get_labels_dict(dataset, labels_version)
    tokenizer = define_tokenizer()
    data_module = JointsluDataModule(dataset=dataset,
                                     labels_version=labels_version,
                                     tokenizer=tokenizer)
    model = LitBertTokenClassification(labels_dict=labels_dict,
                                       tokenizer=tokenizer,
                                       learning_rate=2.9908676527677725e-05)
    log_folder = "v" + str(model_version)
    name = dataset + "_lv" + str(labels_version)
    logger = TensorBoardLogger(log_folder, name=name)
    trainer = Trainer(max_epochs=3, logger=logger)
    trainer.fit(model, datamodule=data_module)


def auto_lr_bz_train(dataset, labels_version):
    """train model with best batch size and learning rate found by lightning"""
    labels_dict = get_labels_dict(dataset, labels_version)
    tokenizer = define_tokenizer()
    model = LitBertTokenClassification(labels_dict=labels_dict,
                                       tokenizer=tokenizer)
    data_module = JointsluDataModule(dataset=dataset,
                                     labels_version=labels_version,
                                     tokenizer=tokenizer)

    logger = TensorBoardLogger("pretrain/auto_sim", name="bert_jointslu")
    trainer = Trainer(auto_lr_find=True,
                      auto_scale_batch_size=True,
                      max_epochs=3,
                      logger=logger)
    trainer.tune(model, datamodule=data_module)
    print(f"lr {model.learning_rate} bz {model.batch_size}")
    trainer.fit(model, datamodule=data_module)


def load_from_checkpoint(tokenizer, model,
                         path="/pretrain/auto_sim/bert_jointslu/version_0/checkpoints/epoch=2-step=11946.ckpt"):

    current_path =str(Path().absolute())
    # path = "/pretrain/auto_sim/bert_jointslu/version_0/checkpoints/epoch=2-step=11946.ckpt"
    chk_path = Path(current_path + path)
    # print("check point path: ", chk_path)
    trained_model = model.load_from_checkpoint(chk_path, tokenizer=tokenizer)
    return trained_model


def predict(batch_size):
    tokenizer = define_tokenizer()
    model = LitBertTokenClassification(tokenizer=tokenizer, batch_size=batch_size)
    data_module = JointsluDataModule(tokenizer=tokenizer)
    # get trained model
    trained_model = load_from_checkpoint(tokenizer, model)
    trainer = Trainer()
    results = trainer.predict(model=trained_model, datamodule=data_module)
    print("results shape ", results)
    print("results=============", results)
    post_processing(results)
    return results


def post_processing(results):
    """data is results from finetuned bert model, generate std_gt and std_prediction and store in file."""
    # results is a list of result:
    #            result = {
    #            "text": input_token,
    #            "gt": gt,
    #            "prediction": prediction}
    post_precessed = []
    from datetime import date
    timestamp = str(date.today())
    for nth_prediction in results:
        for r in nth_prediction:
            text = r["text"]
            gt = r["gt"]
            prediction = r["prediction"]
            std_gt = get_std_gt(text, gt)
            std_output = get_std_gt(text, prediction)
            new_dict = {
                "timestamp": timestamp,
                "text": text,
                "gt": gt,
                "prediction": prediction,
                "std_output": std_output,
                "std_gt": std_gt
            }
            print(new_dict, '\n')
            post_precessed.append(new_dict)
    append_to_json(file_path="../data/jointslu/pre-train/pre-train_outputs/pre-train_output.json", data=post_precessed)


if __name__ == '__main__':
    predict(batch_size=1)
