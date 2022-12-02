import json

from parse.parsing_evaluation import testing

from gpt3.gpt3jointslu import gpt3jointslu
from gpt3.gpt3_util import read_output_from_file
from evaluation.evaluation import evaluate_model, evaluate_acc_f1
from util import read_jointslu_labels


def evaluate_all_labels():
    # read all simplified labels
    f = open("data/jointslu/pre-train/labels.json")
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
    f = open("data/jointslu/pre-train/labels.json")
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
    # evaluate_all_labels()
    # evaluate_all()
    # acc_f1_all()
    acc_f1_all_labels()