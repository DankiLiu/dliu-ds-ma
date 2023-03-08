import csv
from typing import List, Dict

from data.data_processing import get_intents_labels_keys, get_intents_dict, get_labels_dict
from evaluation.evaluation import model_evaluation
from evaluation.metric_score import MetricByModel, MetricByLabel
from evaluation.muc5 import MUC5


def evaluate_bylabel(sample_num, dataset, labels_version, generate, scenario, num_experiments, model):
    if generate:
        model_evaluation(dataset, sample_num, True, labels_version, scenario, model, num_experiments)
    # get labels
    intents_dict = get_intents_dict(dataset, labels_version, scenario)
    labels_dict = get_labels_dict(dataset, labels_version, scenario)
    intents = list(intents_dict.keys())
    labels = list(labels_dict.keys())

    # eval intents
    intents_mucs = calculate_muc_by_labels(intents, dataset, model)
    labels_mucs = calculate_muc_by_labels(labels, dataset, model)

    intent_file_name = store_by_label("intent", dataset, labels_version, sample_num, num_experiments, model)
    write_dicts_to_file(intents_mucs, intent_file_name)

    labels_file_name = store_by_label("label", dataset, labels_version, sample_num, num_experiments, model)
    write_dicts_to_file(labels_mucs, labels_file_name)


def store_by_label(label_type, dataset, labels_version, sample_num, num_experiments, model):
    file_index = 0
    if label_type == "intent":
        folder_name = "evaluation/" + model + "/" + dataset + "/bylabel/intent_mucs_lv"
    else:
        folder_name = "evaluation/" + model + "/" + dataset + "/bylabel/label_mucs_lv"
    file_name = labels_version + "_n" + str(sample_num) \
                + "e" + str(num_experiments) + "_i" + str(file_index) + ".csv"
    import os
    while os.path.exists(folder_name + file_name):
        file_index += 1
        file_name = labels_version + "_n" + str(sample_num) \
                    + "e" + str(num_experiments) + "_i" + str(file_index) + ".csv"
    print("[store_by_labels] file: ", folder_name + file_name)
    return folder_name + file_name


def calculate_muc_by_labels(labels, dataset, model):
    mucs = []
    for label in labels:
        metric = load_metric(dataset, mode="bylabel", label=label, model=model)
        print(label)
        metric.normalization()
        muc = MUC5(metric.cor, metric.par, metric.inc, metric.mis, metric.spu)
        muc_dict = {
            "label": label,
            "acc": muc.acc(),
            "f": muc.f()
        }
        mucs.append(muc_dict)
    return mucs


def load_metric(dataset, mode, model, label=None):
    latest_file = get_latest_files(dataset, mode, model)
    if mode == "bylabel":
        metric = MetricByLabel.create_data_from_file(model_name=model, label_name=label, path=latest_file)
    else:
        metric = MetricByModel.create_data_from_file(model_name=model, path=latest_file)
    print(f"model {metric.model_name}: cor: {metric.cor}, par: {metric.par}, "
          f"inc: {metric.inc}, mis: {metric.mis}, spu: {metric.spu}")
    return metric


def evaluate_bymodel(sample_num, dataset, labels_version, generate, scenario, num_experiments, model):
    # if file not exist, do model_evaluation
    # evaluate model and store the metrics under evaluation folder
    if generate:
        model_evaluation(dataset, sample_num, False, labels_version, scenario, model, num_experiments)
    metric = load_metric(dataset, mode="bymodel", model=model)
    muc = MUC5(metric.cor, metric.par, metric.inc, metric.mis, metric.spu)
    print(f"model {metric.model_name}: \nacc {muc.acc()}, f {muc.f()}")
    # muc_dict[metric.model_name] = muc
    muc_dict = {
        "model": metric.model_name,
        "acc": muc.acc(),
        "f": muc.f()
    }

    file_name = store_by_model(dataset, labels_version, sample_num, num_experiments, model)
    write_dicts_to_file([muc_dict], file_name)


def store_by_model(dataset, labels_version, sample_num, num_experiments, model):
    file_index = 0
    folder_name = "evaluation/" + model + "/" + dataset + "/bymodel/mucs_lv"
    file_name = labels_version + "_n" + str(sample_num) + "e" + str(num_experiments) \
                + "_i" + str(file_index) + ".csv"
    import os
    while os.path.exists(folder_name + file_name):
        file_index += 1
        file_name = labels_version + "_n" + str(sample_num) + "e" + str(num_experiments) \
                    + "_i" + str(file_index) + ".csv"
    print("store by model file name: ", folder_name + file_name)
    return folder_name + file_name


def get_latest_files(dataset, mode, model):
    """get latest metrics files"""
    metrics_folder = "evaluation/" + model + "/" + dataset + "/" + mode + "/"
    # read metrics file and calculate acc and f1
    import glob, os
    list_of_files = glob.glob(metrics_folder + 'e*.csv')
    file = max(list_of_files, key=os.path.getctime)
    return file


def write_dicts_to_file(data: List[Dict], file_name):
    # print("data: ", data)
    with open(file_name, 'w+') as f:
        keys = list(data[0].keys())
        writer = csv.writer(f)
        writer.writerow(keys)
        rows = [list(d.values()) for d in data]
        writer.writerows(rows)
        f.close()
