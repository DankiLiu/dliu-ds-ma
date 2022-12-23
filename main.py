import json

from evaluation.evaluation_utils import merge_data
from evaluation.plotting import plot_model_comparison
from parse.parsing_evaluation import testing

from gpt3.gpt3jointslu import gpt3jointslu
from gpt3.gpt3_util import read_output_from_file
from evaluation.evaluation import evaluate_model, evaluate_acc_f1
from util import read_jointslu_labels, get_gpt3_params, get3output_paths


def run_models(num, parsing_v, pretrain_v, gpt3_v):
    """run all three models with defined model_version and store the results in ***_output.json"""
    # get output files path
    parsing_p, pretrain_p, gpt3_p = get3output_paths(parsing_v, pretrain_v, gpt3_v)
    # get model parameters
    prompt, model_name, select = get_gpt3_params(gpt3_v)
    if prompt is None:
        print(f"model version {gpt3_v} not avaliable")
    gpt3jointslu(num, prompt, model_name, gpt3_p, select)



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
    merge_data("pre-train", "ALL")
