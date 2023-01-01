import json

from data.data_processing import check_training_data
from evaluation.evaluation_utils import merge_data
from parse.parsing_evaluation import testing

from gpt3.gpt3jointslu import gpt3jointslu
from gpt3.gpt3_util import read_output_from_file
from evaluation.evaluation import evaluate_model, evaluate_acc_f1
from util import read_jointslu_labels, get_gpt3_params, get3output_paths, get_output_path


def run_models(num, parsing_v, pretrain_v, gpt3_v, labels_version):
    """run all three models with defined model_version and store the results in ***_output.json"""
    # construct training data and store them into training_data file
    pass


def test_model(model_name, num, model_version, labels_version):
    print(f"1. model_name: {model_name} v{model_version}, lv{labels_version}")
    output_file = get_output_path(model_name=model_name,
                                  model_version=model_version)
    print(f"output will store in file {output_file}")
    # check if training file is available
    if not check_training_data(labels_version):
        print("something wrong with training data")
        return
    print("2. checked training data")
    # run model with name
    if model_name == "gpt3":
        print("3. run gpt3 model")
        run_gpt3_model(num, model_version, output_file, labels_version)


def run_gpt3_model(num, model_version, output_file, labels_version):
    prompt, model_name, select = get_gpt3_params(model_version)
    if prompt is None:
        print(f"model v{model_version} not avaliable, run gpt3 model failed")
        return
    gpt3jointslu(num=num,
                 prompt=prompt,
                 model_name=model_name,
                 path=output_file,
                 select=select,
                 labels_version=labels_version)


def evaluate_all_labels():
    # read all simplified labels
    f = open("data/jointslu/pre-train/labels00.json")
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
    f = open("data/jointslu/pre-train/labels00.json")
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
    test_model("gpt3", 10, 1, "01")
