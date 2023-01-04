from gpt3.gpt3 import gpt3jointslu
from parse.parsing import parse_testing
from pretrain.train import pretrain_testing


def run_gpt3_model(dataset, num, model_version, labels_version, testing_file, output_file):
    """run num of tests on gpt3.
    model_version defines model settings
    dataset amd labels_version defines the data and labels information"""
    gpt3jointslu(dataset=dataset,
                 num=num,
                 model_version=model_version,
                 testing_file=testing_file,
                 output_path=output_file,
                 labels_version=labels_version)


def run_parsing_model(dataset, num, model_version, labels_version, testing_file, output_file):
    """run num of tests on parsing model.
    model_version defines model settings
    dataset amd labels_version defines the data and labels information"""
    parse_testing(testing_file=testing_file,
                  num=num,
                  model_version=model_version,
                  dataset=dataset,
                  output_file=output_file,
                  labels_version=labels_version)


def run_pretrain_model(dataset, model_version, labels_version, output_file):
    """run num of tests on pre-train model.
    model_version defines model hyper-parameters and tokenizer information etc.
    dataset amd labels_version defines the data and labels information, also decides which model to load"""
    pretrain_testing(dataset=dataset,
                     model_version=model_version,
                     labels_version=labels_version,
                     output_file=output_file)
