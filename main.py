import json

from data.data_processing import check_training_data

from gpt3.gpt3 import gpt3jointslu
from evaluation.evaluation import evaluate_model, evaluate_acc_f1
from model_run import run_gpt3_model, run_parsing_model, run_pretrain_model
from util import get_output_path


def test_model(model_name, num, model_version, dataset, labels_version):
    print("==============[ run model testing ]==============")
    print(f"1. {model_name} model-v{model_version}, lv{labels_version}")
    output_file = get_output_path(model_name=model_name,
                                  dataset=dataset,
                                  model_version=model_version)
    print(f"result will be stored in >> {output_file}")
    # check if training file is available
    train_fp, test_fp, val_fp = check_training_data(dataset, labels_version)
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
                       output_file=output_file)
    if model_name == "parsing":
        run_parsing_model(dataset=dataset,
                          num=num,
                          model_version=model_version,
                          labels_version=labels_version,
                          testing_file=test_fp,
                          output_file=output_file)
    if model_name == "pre-train":
        run_pretrain_model(dataset=dataset,
                           model_version=model_version,
                           labels_version=labels_version,
                           output_file=output_file)


def evaluate_all_labels():
    # read all simplified labels
    f = open("data/jointslu/labels/labels00.json")
    labels_dict = json.load(f)
    f.close()
    labels = list(labels_dict.keys())
    for label in labels:
        # for each label, test all 500 examples
        # evaluate_model(model_name="pre-train", num_exps=500, label_name=label)
        evaluate_model(model_name="parsing", num_exps=500, label_name=label)
        evaluate_model(model_name="gpt3", num_exps=500, label_name=label)


def evaluate_all():
    evaluate_model(model_name="pre-train", num_exps=500)
    evaluate_model(model_name="parsing", num_exps=500)
    evaluate_model(model_name="gpt3", num_exps=500)


def acc_f1_all():
    evaluate_acc_f1("gpt3", "ALL", True)
    evaluate_acc_f1("parsing", "ALL", True)
    evaluate_acc_f1("pre-train", "ALL", True)
    evaluate_acc_f1("gpt3", "ALL", False)
    evaluate_acc_f1("parsing", "ALL", False)
    evaluate_acc_f1("pre-train", "ALL", False)


def acc_f1_all_labels():
    # read all simplified labels
    f = open("data/jointslu/labels/labels00.json")
    labels_dict = json.load(f)
    f.close()
    labels = list(labels_dict.keys())
    for label in labels:
        evaluate_acc_f1("gpt3", label, True)
        evaluate_acc_f1("parsing", label, True)
        evaluate_acc_f1("pre-train", label, True)
        evaluate_acc_f1("gpt3", label, False)
        evaluate_acc_f1("parsing", label, False)
        evaluate_acc_f1("pre-train", label, False)


if __name__ == '__main__':
    # test gpt3 pipeline with 10 examples
    test_model(model_name="parsing",
               num=5,
               model_version=0,
               dataset="jointslu",
               labels_version="01")
