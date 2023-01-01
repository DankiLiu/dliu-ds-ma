import os
from json import JSONDecodeError

import requests
from nltk.parse.corenlp import CoreNLPServer
import json


def corenlp_server_start(path_to_jar=None,
                         path_to_models_jar=None):
    """start corenlp server"""
    # Only in testing phase
    path_to_jar = 'C:/Users/liuda/Documents/Files/StudiumInDtl/Masterarbeit/corenlp/stanford-corenlp-4.5.0/stanford-corenlp-4.5.0.jar'
    path_to_models_jar = 'C:/Users/liuda/Documents/Files/StudiumInDtl/Masterarbeit/corenlp/stanford-corenlp-4.5.0/stanford-corenlp-4.5.0-models.jar'
    if not path_to_jar:
        path_to_jar = input("input the path to stanford-corenlp-4.5.0.jar/n")
    if not path_to_models_jar:
        path_to_models_jar = input("input the path to stanford-corenlp-4.5.0-model.jar/n")
    # Server
    server = CoreNLPServer(path_to_jar=path_to_jar,
                           path_to_models_jar=path_to_models_jar)
    print("Starting corenlp server ...")
    server.start()
    return server


def corenlp_server_stop(server):
    """stop corenlp server"""
    server.stop()
    print("Corenlp server stopped...")


def server_is_running(url):
    """return True if corenlp server is running, otherwise False"""
    try:
        page = requests.get(url)
        status_code = page.status_code
    except Exception as e:
        print(e)
        return False
    print(f"status code from {url} is {status_code}")
    if status_code == 200:
        print("server running, ready to parse")
        return True
    else:
        print("server not running")
        return False


def read_jointslu_labels():
    """return a list of BIO labels for jointslu dataset"""
    with open("../data/jointslu.json") as f:
        data = json.load(f)
        jointslu_labels = data["jointslu_labels"]
    return jointslu_labels


def save_jointslu_labels(new_labels):
    """Save all labels appeared in dataset into file
    :paras new_labels: a list of label lists"""
    data = {"jointslu_labels": list(new_labels)}
    print("label set", data)
    with open("../data/jointslu.json", 'w') as f:
        json.dump(data, f, indent=4)


def append_to_json(file_path, new_data):
    """append new data to current json file. data is a list of json dicts"""
    f = open(file_path, 'r')
    data = []
    if os.path.getsize(file_path) > 0:
        data = json.load(f)
    else:
        print(f"{file_path} file is empty")
    f.close()
    data.append(new_data)
    f = open(file_path, 'w')
    json.dump(data, f, indent=2)
    f.close()


def read_from_json(path):
    """read and return the json file content"""
    file = open(path, 'r')
    data = json.load(file)
    file.close()
    return data


def read_keys_from_json(path, *args):
    """return content of keys (args) of json file"""
    data = read_from_json(path)
    values = {}
    for arg in args:
        try:
            values[arg] = [item[arg] for item in data]
            # print("add item to values: ", values)
        except KeyError:
            print(f"Missing key {arg} in {path} file")
    return values


def get_gpt3_params(version):
    f = open("model_version.json")
    gpt3_data = json.load(f)["gpt3"]
    for item in gpt3_data:
        if item["version"] == version:
            return item["prompt"], item["model"], item["select"]
    return None, None, None


def get_parsing_params(version):
    f = open("model_version.json")
    parsing_data = json.load(f)["parsing"]
    for item in parsing_data:
        if item["version"] == version:
            return item["shuffle"]
    return None


def get3output_paths(parsing_v, pretrain_v, gpt3_v):
    return get_output_path("parsing", parsing_v), \
           get_output_path("pre-train", pretrain_v), \
           get_output_path("gpt3", gpt3_v)


def get_output_path(model_name, model_version):
    """return output path given model name and version,
    model versions are defined in model_version.json file"""
    return create_output_file(model_name, model_version)


def create_output_file(model, v):
    """create or check file existance according to version and return file location
    """
    path = "data/jointslu/"
    if model == "parsing":
        path = path + "parsing/parsing_outputs/"
    elif model == "pre-train":
        path = path + "pre-train/pre-train_outputs/"
    elif model == "gpt3":
        path = path + "gpt3/gpt3_outputs/"
    else:
        print(f"wrong model name [{model}] is given")
        return
    # create file name by version
    from datetime import datetime
    now = datetime.now()  # current date and time
    day = now.strftime("%d")
    month = now.strftime("%m")
    file_name = month + day + 'v' + str(v) + '.json'

    output_path = path + file_name
    from os.path import exists
    file_index = 1
    while exists(output_path):
        print(f"path already exists {output_path}")
        file_name = month + day + 'v' + str(v) + '_' + str(file_index) + '.json'
        output_path = path + file_name
        file_index = file_index + 1
    open(output_path, 'a+').flush()
    print(f"{output_path} created")
    return output_path
