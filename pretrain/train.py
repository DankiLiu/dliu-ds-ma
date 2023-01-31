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
from util import append_to_json, get_pretrain_params, find_ckpt_in_dir


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


def select_data_module(bsz, dataset, labels_version, tokenizer):
    """select different data module, for now jointslu is the only dataset"""
    return JointsluDataModule(dataset=dataset,
                              labels_version=labels_version,
                              tokenizer=tokenizer)


def train(model_version, dataset, labels_version):
    lr, max_epoch, batch_size = get_pretrain_params(model_version)
    labels_dict = get_labels_dict(dataset, labels_version)
    tokenizer = define_tokenizer()
    data_module = select_data_module(batch_size, dataset, labels_version, tokenizer)
    model = LitBertTokenClassification(labels_dict=labels_dict,
                                       tokenizer=tokenizer,
                                       learning_rate=lr)
    log_folder = "v" + str(model_version)
    name = dataset + "_lv" + str(labels_version)
    logger = TensorBoardLogger(log_folder, name=name)
    trainer = Trainer(max_epochs=max_epoch, logger=logger)
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


def load_from_checkpoint(tokenizer, model, ckpt_path):
    current_path = str(Path().absolute())
    chk_path = Path(current_path + ckpt_path)
    # print("check point path: ", chk_path)
    trained_model = model.load_from_checkpoint(chk_path, tokenizer=tokenizer)
    return trained_model


def pretrain_testing(dataset, model_version, labels_version, output_file):
    # todo: does pretrain model have to predict all examples from datamodule?
    # get pre-train model parameters
    lr, max_epoch, batch_size = get_pretrain_params(model_version)
    print(f"    [pretrain_testing] v{model_version} lr={lr}, batch_size={batch_size}, max_epoch={max_epoch}")
    # get prediction results
    results = pretrain_predict(dataset, model_version, labels_version, lr, batch_size)
    # post processing and store output in file
    post_processing(results, output_file)


def pretrain_predict(dataset, model_version, labels_version, lr, batch_size):
    tokenizer = define_tokenizer()
    labels_dict = get_labels_dict(dataset=dataset, labels_version=labels_version)
    model = LitBertTokenClassification(tokenizer=tokenizer,
                                       labels_dict=labels_dict,
                                       learning_rate=lr,
                                       batch_size=batch_size)
    # select data module
    data_module = select_data_module(batch_size, dataset, labels_version, tokenizer)
    # construct path with model_version and labels_version
    log_folder = "v" + str(model_version)
    name = dataset + "_lv" + str(labels_version)
    model_path = log_folder + "/" + name
    # find .ckpt file in folder
    ckpt_file = find_ckpt_in_dir(model_path)
    if ckpt_file is None:
        print("no checkpoint available")
        return
    print(f"    [pretrain_testing] load model from {ckpt_file}")
    trained_model = load_from_checkpoint(tokenizer, model, ckpt_file)
    trainer = Trainer()
    results = trainer.predict(model=trained_model, datamodule=data_module)
    print("    [pretrain_testing] results shape is ", results)
    return results


def post_processing(results, output_file):
    """data is results from finetuned bert model, generate std_gt and std_prediction and store in file."""
    # results is a list of result:
    #            result = {
    #            "text": input_token,
    #            "gt": gt,
    #            "prediction": prediction}
    post_precessed = []
    for nth_prediction in results:
        for r in nth_prediction:
            text = r["text"]
            gt = r["gt"]
            prediction = r["prediction"]
            std_gt = get_std_gt(text, gt, None)
            std_output = get_std_gt(text, prediction, None)
            new_dict = {
                "num": nth_prediction,
                "text": text,
                "gt": gt,
                "prediction": prediction,
                "std_output": std_output,
                "std_gt": std_gt
            }
            post_precessed.append(new_dict)
    append_to_json(file_path=output_file, new_data=post_precessed)
    print(f"{len(post_precessed)} results appended to parsing_output.json")

