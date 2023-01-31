import json

from data.data_processing import check_training_data, check_labels_atis
from evaluation.evaluation import evaluate_acc_f1, model_evaluation
from evaluation.main import evaluate_bymodel, load_bymodel_metrics, evaluate_bylabel
from model_run import run_gpt3_model, run_parsing_model, run_pretrain_model
from pretrain.multi_task.main import train_multi_task, mt_testing
from pretrain.train import train
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
    if dataset == "jointslu":
        # for jointslu dataset, check the labels and intents files
        check_labels_atis(labels_version)
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


if __name__ == '__main__':
    # train pretrained model with new label set
    # evaluate model and store the metrics under evaluation folder
    # load_bymodel_metrics("jointslu")
    # model_evaluation(dataset="jointslu", sample_num=500,
    #                 bylabel=True, labels_version="01")
    evaluate_bylabel(500, "jointslu", "01", False)