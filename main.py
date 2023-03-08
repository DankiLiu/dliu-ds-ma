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
                           scenario=scenario,
                           few_shot_num=few_shot_num)


def fintune_bert_from_ckpt(dataset, labels_version, model_version, few_shot_num, scenario):
    train_fp, test_fp, val_fp = check_training_data(dataset, labels_version, scenario, few_shot_num)
    if train_fp is None or test_fp is None or val_fp is None:
        print("data files are missing")
        return
    old_tasks = define_tasks(dataset="jointslu", labels_version="01", scenario=None)
    config = Config(classifier_only=True, from_ckpt=True, auto_lr=True, old_task=old_tasks, few_shot_num=few_shot_num)
    # if train from checkpoint, train from model_version
    train_multi_task(model_version=model_version, dataset="massive",
                     labels_version="00", scenario=scenario, config=config)


def test_three_model(parsing, pretrain, gpt3, num, scenario):
    # parsing
    if parsing:
        test_model(model_name="parsing", num=50, model_version=0,
                   dataset="massive", labels_version="00", scenario=scenario, few_shot_num=-1)
    # pre-train
    if pretrain:
        test_model(model_name="pre-train", num=50, model_version=5,
                   dataset="massive", labels_version="00", scenario=scenario, few_shot_num=100)
    # gpt3
    if gpt3:
        test_model(model_name="gpt3", num=50, model_version=1,
                   dataset="massive", labels_version="00", scenario=scenario, few_shot_num=-1)


def evaluate(generate=True):
    sample_num = 50
    dataset = "massive"
    labels_version = "00"
    scenario = "alarm"
    evaluate_bymodel(sample_num, dataset, labels_version, generate, scenario)


if __name__ == '__main__':
    if False:
        fintune_bert_from_ckpt(dataset="massive",
                               labels_version="00",
                               model_version=1,
                               few_shot_num=-1,
                               scenario="weather")
    # exit()
    # test_three_model()

    # evaluate()
    generate_ori_with_scenario("news")