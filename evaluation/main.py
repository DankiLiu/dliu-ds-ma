from data.data_processing import get_intents_labels_keys
from evaluation.evaluation import model_evaluation
from evaluation.metric_score import MetricByModel, MetricByLabel
from evaluation.muc5 import MUC5


def evaluate_bylabel(sample_num, dataset, labels_version, generate):
    if generate:
        model_evaluation(dataset, sample_num, True, labels_version)
    # get labels
    labels = get_intents_labels_keys(dataset, labels_version)
    print(len(labels), " ", labels)
    muc_dict = {}
    for label in labels:
        metrics = load_bylabel_metrics(dataset, label)
        print(label)
        for metric in metrics:
            metric.normalization()
            muc = MUC5(metric.cor, metric.par, metric.inc, metric.mis, metric.spu)

            print(f"    {muc.acc()} ; {muc.f()}")


def load_bylabel_metrics(dataset, label):
    model_list = ["parsing", "pre-train", "gpt3"]
    latest_file = get_latest_file(dataset, "bylabel")
    metrics = []
    if latest_file:
        # create a metric class with file
        for model_name in model_list:
            metric = MetricByLabel.create_data_from_file(model_name=model_name, label_name=label, path=latest_file)
            # print(f"model {metric.model_name}: cor: {metric.cor}, par: {metric.par}, inc: {metric.inc}, mis: {metric.mis}, spu: {metric.spu}")
            metrics.append(metric)
    return metrics


def evaluate_bymodel(sample_num, dataset, labels_version, generate: bool):
    # if file not exist, do model_evaluation
    # evaluate model and store the metrics under evaluation folder
    if generate:
        model_evaluation(dataset, sample_num, False, labels_version)
    metrics = load_bymodel_metrics(dataset)
    muc_dict = {}

    for metric in metrics:
        muc = MUC5(metric.cor, metric.par, metric.inc, metric.mis, metric.spu)
        print(f"model {metric.model_name}: \nacc {muc.acc()}, f {muc.f()}")
        # muc_dict[metric.model_name] = muc


def load_bymodel_metrics(dataset):
    model_list = ["parsing", "pre-train", "gpt3"]
    latest_file = get_latest_file(dataset, "bymodel")
    metrics = []
    if latest_file:
        # create a metric class with file
        for model_name in model_list:
            metric = MetricByModel.create_data_from_file(model_name=model_name, path=latest_file)
            print(f"model {metric.model_name}: cor: {metric.cor}, par: {metric.par}, inc: {metric.inc}, mis: {metric.mis}, spu: {metric.spu}")
            metrics.append(metric)
    return metrics


def get_latest_file(dataset, mode):
    metrics_folder = "evaluation/" + dataset + "/" + mode + "/"
    # read metrics file and calculate acc and f1
    import glob, os
    latest_file = ""
    list_of_files = glob.glob(metrics_folder + '*.csv')
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file