from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import BertTokenizer

from data.data_processing import get_labels_dict, get_intents_dict
from evaluation.evaluation_utils import get_std_gt
from pretrain.multi_task.multi_task_lightning import LitBertMultiTask
from pretrain.multi_task.jointslu_mt_data_module import MTDataModule
from pretrain.multi_task.multi_task_bert import Task
from util import get_pretrain_params, get_pretrain_checkpoint, find_ckpt_in_dir, append_to_json

import numpy as np


class Config:
    def __init__(self, from_ckpt, classifier_only, auto_lr, old_task, few_shot):
        self.from_ckpt = from_ckpt
        self.classifier_only = classifier_only
        self.auto_lr = auto_lr
        if self.from_ckpt:
            self.old_task = old_task
        self.few_shot = few_shot


def train_multi_task(model_version, dataset, labels_version, scenario, config):
    # define tasks
    tasks = define_tasks(dataset, labels_version, scenario)
    # init tokenizer
    tokenizer = define_tokenizer()
    # create data module
    data_module = MTDataModule(dataset=dataset,
                               labels_version=labels_version,
                               scenario=scenario,
                               tokenizer=tokenizer,
                               few_shot=config.few_shot)
    model = None
    if not config.from_ckpt:
        model = LitBertMultiTask(tokenizer=tokenizer, tasks=tasks, classifier_only=config.classifier_only)
        print(f"multi_task classifier layer hidden size: {model.multi_task_bert.output_heads['0'].hidden_size}, "
          f"{model.multi_task_bert.output_heads['0'].num_labels}, {model.multi_task_bert.output_heads['1'].num_labels}")

    if config.from_ckpt:  # load model from checkpoint
        print("load from ckpt")
        trained_model = LitBertMultiTask(tokenizer=tokenizer,
                                         tasks=config.old_task,
                                         classifier_only=config.classifier_only)
        require_grad = None
        for param in trained_model.multi_task_bert.encoder.parameters():
            require_grad = param.requires_grad
            break
        print("trained_model requires_grad", require_grad)

        ckpt_file = get_ckpt_file(model_version)
        model = trained_model.load_from_checkpoint(ckpt_file,
                                                   tokenizer=tokenizer,
                                                   tasks=config.old_task,
                                                   classifier_only=config.classifier_only)

        require_grad2 = None
        for param in model.multi_task_bert.encoder.parameters():
            require_grad2 = param.requires_grad
            break
        print("loaded|_model requires_grad", require_grad2)

        print(f"new heads hidden size and labels number: {model.multi_task_bert.output_heads['0'].hidden_size}, "
              f"{model.multi_task_bert.output_heads['0'].num_labels}, "
              f"{model.multi_task_bert.output_heads['1'].num_labels}")

        # todo: set classification layer to new task settings
        model.multi_task_bert.set_output_heads(tasks)
        require_grad_e, require_grad_d = None, None
        for param in model.multi_task_bert.encoder.parameters():
            require_grad_e = param.requires_grad
            break
        for param in model.multi_task_bert.output_heads['0'].parameters():
            require_grad_e = param.requires_grad
            break
        print("rehead_model encoder required_grad: ", require_grad_e)
        print("rehead_model decoder required_grad: ", require_grad_d)
        print(f"new heads hidden size and labels number: {model.multi_task_bert.output_heads['0'].hidden_size}, "
              f"{model.multi_task_bert.output_heads['0'].num_labels}, "
              f"{model.multi_task_bert.output_heads['1'].num_labels}")

    # log folder naming
    print("set log files")
    log_folder = "pretrain/mt_v" + str(model_version)
    name = dataset + "_lv" + str(labels_version)
    if scenario:
        name += "_" + scenario
    if config.classifier_only:
        name += "_cl"
    if config.few_shot:
        name += "_few"
    logger = TensorBoardLogger(log_folder, name=name)

    lr, max_epoch, batch_size = get_pretrain_params(model_version)

    # todo: print some info about the loaded model

    if config.auto_lr:
        print("auto lr")
        """
        trainer = Trainer()
        lr_finder = trainer.tuner.lr_find(model)

        print(lr_finder.results)
        fig = lr_finder.plot(suggest=True)
        fig.show()

        new_lr = lr_finder.suggestion()
        model.hparams.lr = new_lr
        print(model.hparams)
        trainer.tune(model, datamodule=data_module)
        print(f"lr {model.learning_rate} bz {model.batch_size}")
        exit()
        """
        trainer = Trainer(auto_lr_find=True,
                          auto_scale_batch_size=True,
                          max_epochs=3,
                          logger=logger)
        trainer.fit(model, datamodule=data_module)
    else:
        trainer = Trainer(max_epochs=1, logger=logger)
        trainer.fit(model, datamodule=data_module)


def define_tasks(dataset, labels_version, scenario):
    # define tasks
    labels_dict = get_labels_dict(dataset, labels_version, scenario)
    intents_dict = get_intents_dict(dataset, labels_version, scenario)

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


def mt_testing(dataset, model_version, labels_version, output_file, scenario):
    # define tasks (tok and seq classification)
    tasks = define_tasks(dataset, labels_version, scenario)
    # init tokenizer
    tokenizer = define_tokenizer()
    # init data module
    # todo: load massive data
    data_module = MTDataModule(dataset=dataset,
                               labels_version=labels_version,
                               scenario=scenario,
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
    # todo: if dataset is massive, any difference?
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
