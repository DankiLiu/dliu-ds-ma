import json

from data.data_processing import check_training_data, check_labels_atis
from data.massive.data_processing import generate_ori_with_scenario, check_labels_massive
from evaluation.evaluation import model_evaluation
from evaluation.main import evaluate_bymodel

from model_run import run_gpt3_model, run_parsing_model, run_pretrain_model
from pretrain.multi_task.main import Config, train_multi_task, define_tasks
from util import get_output_path


def test_model(model_name, num, model_version, dataset, labels_version, scenario, few_shot_num=-1):
    print("==============[ run model testing ]==============")
    print(f"1. {model_name} model-v{model_version}, lv{labels_version}")
    output_file = get_output_path(model_name=model_name,
                                  dataset=dataset,
                                  model_version=model_version,
                                  scenario=scenario)
    print(f"result will be stored in >> {output_file}")

    if dataset == "jointslu":
        # for jointslu dataset, check the labels and intents files
        check_labels_atis(labels_version)
    elif dataset == "massive":
        check_labels_massive(labels_version, scenario)

    train_fp, test_fp, val_fp = check_training_data(dataset, labels_version, scenario, few_shot_num)
    if train_fp is None or test_fp is None or val_fp is None:
        print("data files are missing")
        return

    print("2. got data paths")
    # todo: should labels files be checked here? because labels and training data
    #  are all determined by model_version, dataset and labels_version information
    # run model with name
    if model_name == "gpt3":
        print("3. run gpt3 model with testing data file")
        run_gpt3_model(dataset=dataset,
                       num=num,
                       model_version=model_version,
                       labels_version=labels_version,
                       testing_file=test_fp,
                       output_file=output_file,
                       scenario=scenario)
    if model_name == "parsing":
        run_parsing_model(dataset=dataset,
                          num=num,
                          model_version=model_version,
                          labels_version=labels_version,
                          testing_file=test_fp,
                          output_file=output_file,
                          scenario=scenario)
    if model_name == "pre-train":
        run_pretrain_model(dataset=dataset,
                           model_version=model_version,
                           labels_version=labels_version,
                           output_file=output_file,
                           scenario=scenario)


def fintune_bert_from_ckpt(dataset, labels_version, model_version, few_shot_num, scenario):
    train_fp, test_fp, val_fp = check_training_data(dataset, labels_version, scenario, few_shot_num)
    if train_fp is None or test_fp is None or val_fp is None:
        print("data files are missing")
        return
    old_tasks = define_tasks(dataset="jointslu", labels_version="02", scenario=None)
    config = Config(classifier_only=True,
                    from_ckpt=True,
                    auto_lr=True,
                    old_task=old_tasks,
                    few_shot_num=few_shot_num,
                    early_stopping=True,
                    epoch=300,
                    auto_batch_size=True)
    # if train from checkpoint, train from model_version
    train_multi_task(model_version=model_version, dataset="massive",
                     labels_version="00", scenario=scenario, config=config)


def fintune_bert(dataset, labels_version, model_version, few_shot_num, scenario, early_stopping):
    train_fp, test_fp, val_fp = check_training_data(dataset, labels_version, scenario, few_shot_num)
    if train_fp is None or test_fp is None or val_fp is None:
        print("data files are missing")
        return
    config = Config(classifier_only=True,
                    from_ckpt=False,
                    auto_lr=True,
                    old_task=None,
                    few_shot_num=few_shot_num,
                    early_stopping=early_stopping,
                    epoch=300,
                    auto_batch_size=True)
    # if train from checkpoint, train from model_version
    train_multi_task(model_version=model_version, dataset=dataset,
                     labels_version=labels_version, scenario=scenario, config=config)


def few_shot_from_v2():
    # store in mt_v7
    # model_versions = [7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7]
    num = [-1, 12, 20, 30, 50, 100, 200, 300]
    for n in num:
        for i in range(3):
            fintune_bert_from_ckpt(dataset="massive",
                                   labels_version="00",
                                   model_version=2.0,
                                   few_shot_num=n,
                                   scenario="alarm")


def few_shot():
    # store in mt_v6
    model_versions = [6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7]
    num = [-1, 12, 20, 30, 50, 100, 200, 300]
    for n in range(len(model_versions)):
        for i in range(3):
            fintune_bert(dataset="massive",
                         labels_version="00",
                         model_version=model_versions[n],
                         few_shot_num=num[n],
                         scenario="alarm",
                         early_stopping=True)


if __name__ == '__main__':
    #scenarios = ["alarm", "audio", "iot", "music", "news", "takeaway", "weather"]
    scenarios = ["alarm"]
    if False:
        for scenario in scenarios:
            fintune_bert_from_ckpt(dataset="massive",
                                   labels_version="00",
                                   model_version=2.0,
                                   few_shot_num=-1,
                                   scenario=scenario)
    if False:
        for scenario in scenarios:
            fintune_bert(dataset="massive",
                         labels_version="00",
                         model_version=3.1,
                         few_shot_num=-1,
                         scenario=scenario,
                         early_stopping=True)

    if False:
        few_shot = [300]
        for num in few_shot:
            fintune_bert_from_ckpt(dataset="massive",
                                   labels_version="00",
                                   model_version=2.0,
                                   few_shot_num=num,
                                   scenario="alarm")
    if False:
        for i in range(3):
            fintune_bert(dataset="massive",
                         labels_version="00",
                         model_version=6.0,
                         few_shot_num=300,
                         scenario="alarm",
                         early_stopping=True)
        for i in range(3):
            fintune_bert(dataset="massive",
                         labels_version="00",
                         model_version=6.1,
                         few_shot_num=-1,
                         scenario="alarm",
                         early_stopping=True)
    few_shot_from_v2()
    few_shot()