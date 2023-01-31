from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import BertTokenizer

from data.data_processing import get_labels_dict, get_intents_dict
from evaluation.evaluation_utils import get_std_gt
from pretrain.multi_task.multi_task_lightning import LitBertMultiTask
from pretrain.multi_task.jointslu_mt_data_module import JointsluMTDataModule
from pretrain.multi_task.multi_task_bert import Task
from util import get_pretrain_params, get_pretrain_checkpoint, find_ckpt_in_dir, append_to_json

import numpy as np

def train_multi_task(model_version, dataset, labels_version):
    # define tasks
    tasks = define_tasks(dataset, labels_version)
    # init tokenizer
    tokenizer = define_tokenizer()
    # create data module
    data_module = JointsluMTDataModule(dataset=dataset,
                                       labels_version=labels_version,
                                       tokenizer=tokenizer)

    model = LitBertMultiTask(tokenizer=tokenizer, tasks=tasks)
    log_folder = "pretrain/mt_v" + str(model_version)
    name = dataset + "_lv" + str(labels_version)
    # todo: store this file path to model_version file
    logger = TensorBoardLogger(log_folder, name=name)

    lr, max_epoch, batch_size = get_pretrain_params(model_version)
    trainer = Trainer(max_epochs=max_epoch, logger=logger)
    trainer.fit(model, datamodule=data_module)


def define_tasks(dataset, labels_version):
    # define tasks
    labels_dict = get_labels_dict(dataset, labels_version)
    intents_dict = get_intents_dict(dataset, labels_version)

    tok_cls_task = Task(task_id=0,
                        task_name='tok',
                        labels_dict=labels_dict,
                        type='tok_classification')
    seq_cls_task = Task(task_id=1,
                        task_name='seq',
                        labels_dict=intents_dict,
                        type='seq_classification')
    return [tok_cls_task, seq_cls_task]


def define_tokenizer():
    return BertTokenizer.from_pretrained(
        'bert-base-uncased',
        do_lower_case=True,
        sep_token='EOS',
        cls_token='BOS')


def mt_testing(dataset, model_version, labels_version, output_file):
    # define tasks (tok and seq classification)
    tasks = define_tasks(dataset, labels_version)
    # init tokenizer
    tokenizer = define_tokenizer()
    # init data module
    data_module = JointsluMTDataModule(dataset=dataset,
                                       labels_version=labels_version,
                                       tokenizer=tokenizer)
    model = LitBertMultiTask(tokenizer=tokenizer, tasks=tasks)
    # load model from checkpoint
    ckpt_file = get_ckpt_file(model_version)
    model_from_checkpoint = model.load_from_checkpoint(ckpt_file, tokenizer=tokenizer, tasks=tasks)
    # model prediction
    trainer = Trainer()
    predictions = trainer.predict(model=model_from_checkpoint, datamodule=data_module)
    results = post_processing(predictions)
    # save the results in output file
    append_to_json(file_path=output_file, new_data=results)
    print(f"    [pre-trained bert] {len(results)} results appended to parsing_output.json")


def post_processing(predictions):
    """ process the prediction of pre-trained model to the same format as other two approaches
            result = {
            "input_tokens": input_tokens,
            "intent_gts": intent_gts,
            "labels_gts": labels_gts,
            "tok_predictions": tok_predictions,
            "seq_predictions": seq_predictions
        }
    """
    results = []
    print(type(predictions))
    pd_len = len(predictions)
    input_tokens = np.array([predictions[i]["input_tokens"] for i in range(pd_len)]).reshape(pd_len)
    labels_gts = np.array([predictions[i]["labels_gts"] for i in range(pd_len)]).reshape(pd_len)
    intent_gts = np.array([predictions[i]["intent_gts"] for i in range(pd_len)]).reshape(pd_len)
    tok_pre = np.array([predictions[i]["tok_predictions"] for i in range(pd_len)]).reshape(pd_len)
    seq_pre = np.array([predictions[i]["seq_predictions"] for i in range(pd_len)]).reshape(pd_len)
    sample_num = len(input_tokens)
    std_output = [get_std_gt(input_tokens[i], tok_pre[i], seq_pre[i]) for i in range(sample_num)]
    std_gt = [get_std_gt(input_tokens[i], labels_gts[i], intent_gts[i]) for i in range(sample_num)]
    for i in range(sample_num):
        result = {
            "num": i,
            "text": input_tokens[i],
            "intent_gt": intent_gts[i],
            "intent_prediction": seq_pre[i],
            "gt": labels_gts[i],
            "prediction": tok_pre[i],
            "std_output": std_output[i],
            "std_gt": std_gt[i]
        }
        results.append(result)
    return results


def get_ckpt_file(model_version):
    # get ckpt file given model version
    # by pretrained bert model, do not need dataset and labels_version, determined by model version
    checkpoint = get_pretrain_checkpoint(model_version)
    current_path = str(Path().absolute())
    chk_path = Path(current_path + checkpoint)
    ckpt_file = find_ckpt_in_dir(chk_path)
    # find .ckpt file in folder
    if ckpt_file is None:
        print(" [load_from_checkpoint] ckpt_file does not exist")
    return ckpt_file
