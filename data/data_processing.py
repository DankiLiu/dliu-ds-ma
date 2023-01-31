import json
import os
from typing import List

import pandas as pd
from pandas.core.indexing import IndexingError

import util

"""@@@ this is the start of new data processing"""


def iob2json_with_intent(train, test, val):
    """process original iob data into json files,
    input is a list of iob file path, in the order of train, test, val
    json file has key id, intent, text, label, json files are stored in training_data"""
    # get original data
    f_train = open(train, 'r')
    f_test = open(test, 'r')
    f_val = open(val, 'r')
    data_train = f_train.readlines()
    data_test = f_test.readlines()
    data_val = f_val.readlines()
    train_list, test_list, val_list = [], [], []

    # lines to dict
    train_list = lines_to_dict(data_train)
    test_list = lines_to_dict(data_test)
    val_list = lines_to_dict(data_val)

    # store to json file
    train_out = open("data/jointslu/training_data/train.json", 'w+')
    test_out = open("data/jointslu/training_data/test.json", 'w+')
    val_out = open("data/jointslu/training_data/val.json", 'w+')
    json.dump(train_list, train_out, indent=4)
    json.dump(test_list, test_out, indent=4)
    json.dump(val_list, val_out, indent=4)


def lines_to_dict(lines):
    json_list = []
    for i, train_l in enumerate(lines):
        line_split = train_l.split('\t')
        text, label = line_split[0].strip().split(' '), line_split[1].strip().split(' ')
        if len(text) == len(label):
            intent = label[-1]
            label[0] = "[CLS]"
            label[-1] = "[SEP]"
            json_list.append({
                "id": i,
                "intent": intent,
                "text": text,
                "labels": label
            })
    return json_list


def check_labels_atis(labels_version):
    """check existence of intent and label files for atis dataset given labels_version"""
    folder = "data/jointslu/labels/"
    intent_csv = folder + "intents" + labels_version + ".csv"
    intent_json = folder + "intents" + labels_version + ".json"
    labels_csv = folder + "labels" + labels_version + ".csv"
    labels_json = folder + "labels" + labels_version + ".json"
    if os.path.exists(intent_csv) and not os.path.exists(intent_json):
        labels_csv_to_json(intent_csv)
    if os.path.exists(labels_csv) and not os.path.exists(labels_json):
        labels_csv_to_json(labels_csv)


def traversal_intent(file_path):
    """traversal the intents exist in file_path as store it as a column in a csv_file"""
    f = open(file_path, 'r')
    data = json.load(f)
    intents = list(set([item["intent"] for item in data]))
    print(f"found {len(intents)} intents: \n{intents}")
    df = pd.DataFrame({"ori": intents})
    df.to_csv('data/jointslu/labels/intents01.csv', index=False)


def labels_csv_to_json(file_path):
    """store simplified labels and intents as a dictionary to json file, each matches a index."""
    # read from intent file
    df = pd.read_csv(file_path)
    sim_intents = list(set(df['sim'].tolist()))
    intent_dict = dict(zip(sim_intents, range(len(sim_intents))))
    f = open(file_path.split('.')[0] + '.json', 'w+')
    json.dump(intent_dict, f, indent=4)


"""@@@ this is the end of new data processing"""


def read_jointslu_lines(file_path=None):
    if not file_path:
        file_path = "../data/sample.iob"
    with open(file_path) as f:
        lines = f.readlines()
    return lines


def jointslu_per_line(line: str):
    """
    process data from JointSLU dataset
    :param line: one line of data read from .iob file
           path: path
    :return: sentence: a string without EOS and BOS
            words: a list of words without EOS and BOS
            labels: a list of labels, BIO tags
    words and labels should have same length (one to one match)
    """
    # Split the sentence and labels by tab
    sentence, labels = line.split('\t')
    sentence = sentence.strip()
    labels = labels.strip()
    # Strip BOS, EOS labels, first and last labels
    words_list, labels_list = sentence.split(' '), labels.split(' ')
    words_len, labels_len = len(words_list), len(labels_list)
    assert words_len == labels_len

    words = words_list[1: words_len - 1]
    labels = labels_list[1: labels_len - 1]
    sentence = " ".join(words)
    return sentence, words, labels


def find_all_labels(llist, boi=False):
    import itertools
    labels = list(itertools.chain(*llist))
    labels = list(filter(lambda l: l != 'O', labels))

    if not boi:
        labels = [label.split('-')[1] for label in labels]
    labels = set(labels)
    print(f"{len(labels)} labels: {labels}")
    util.save_jointslu_labels(labels)


def store_jointslu_labels():
    with open("jointslu/training_data/train.json") as f:
        data = json.load(f)
    labels_set = []
    print(len(data), " data in train.json")
    for ele in data:
        labels = ele["labels"]
        for label in labels.split(' '):
            labels_set.append(label)
    labels_set = set(labels_set)
    print(f"{len(labels_set)} labels are [{labels_set}]")
    util.save_jointslu_labels(labels_set)


def construct_jointslu_data(type_name: str, lines):
    """Construct the raw data for training usage.
    type: train/test/val
    lines: data lines from jointslu dataset
    train.iob (raw data) into train.json"""

    # Construct raw data into json format
    def strip_extra(old_str):
        new_str = old_str.strip()
        new_str = new_str.strip('\n')
        return new_str

    data = []
    for i in range(len(lines)):
        text, labels = lines[i].split('\t')
        new_text = strip_extra(text)
        new_labels = strip_extra(labels)
        if new_text and new_labels is not None:
            print(f"text length {len(new_text.split(' '))}")
            print(f"labels length {len(new_labels.split(' '))}")
            print(new_text.split(' '))
            print(new_labels.split(' '))
            assert len(new_text.split(' ')) == len(new_labels.split(' '))
            data.append({"id": i + 1,
                         "text": new_text,
                         "labels": new_labels})
    # Store data by type
    store_path = "../data/jointslu/" + type_name + ".json"
    outfile = open(store_path, 'w')
    json.dump(data, outfile, indent=4)


def set_cls_sep_tokens():
    with open("jointslu/temp/val.json") as f:
        data = json.load(f)
    new_data = []
    for dic in data:
        labels = dic["labels"].split(' ')
        labels[0] = "[CLS]"
        labels[-1] = "[SEP]"
        new_labels = ' '.join(labels)
        dic["labels"] = new_labels
        new_data.append(dic)
    f.close()
    outfile = open("jointslu/temp/val.json", 'w')
    print(data[0])
    print(new_data[0])
    json.dump(new_data, outfile, indent=4)
    outfile.close()


def get_intents_labels_keys(dataset, labels_version):
    """return intents and labels for dataset and labels_version with a list"""
    label_dict = get_labels_dict(dataset=dataset, labels_version=labels_version)
    intent_dict = get_intents_dict(dataset=dataset, labels_version=labels_version)
    labels = list(label_dict.keys())
    intents = list(intent_dict.keys())
    return labels + intents


def get_intents_dict(dataset, labels_version):
    """return intents in dictionary form, one intent matches one indenx number"""
    file_path = "data/" + dataset + "/labels/intents" + labels_version + ".json"
    if os.path.exists(file_path):
        f = open(file_path)
        intents_dict = json.load(f)
        f.close()
        print("[get_labels_dict] - ", intents_dict)
        return intents_dict
    return None


def get_labels_dict(dataset, labels_version):
    """return labels in dictionary form, one label matches to one index number"""
    file_path = "data/" + dataset + "/labels/labels" + labels_version + ".json"
    if os.path.exists(file_path):
        f = open(file_path)
        labels_dict = json.load(f)
        f.close()
        print("[get_labels_dict] - ", labels_dict)
        return labels_dict
    return None


def get_ori_sim_dict(dict_type, dataset, labels_version):
    """return ori_sim labels in dictionary form, one sim label matches to one ori label"""
    print(f"getting {dict_type} ori-sim dict")
    file_name = dict_type + labels_version + ".csv"
    ori_sim_labels_path = "data/" + dataset + "/labels/" + file_name
    if os.path.exists(ori_sim_labels_path):
        df = pd.read_csv(ori_sim_labels_path)
        ori, sim = df['ori'].tolist(), df['sim'].tolist()
        ori_sim = dict(zip(ori, sim))
        return ori_sim
    else:
        raise FileNotFoundError(ori_sim_labels_path, "dose not exist, can not proceed")


def gpt3_from_bert_dataset():
    f = open("jointslu/temp/train.json")
    data = json.load(f)
    output_file = open("../data/jointslu/gpt3/train.json", 'a')
    for example in data:
        text = example["text"]
        text = text.replace('EOS', '')
        text = text.replace('BOS', '')
        text = text.strip() + '.'
        print(text)
        output_file.write(text + '\n')
    output_file.close()
    f.close()


def create_training_data(shuffle=False):
    # todo: parsing use same training data as other two models but has preprocessing step
    """create data for parsing from "bert_***.json" without bos and eos
    and store them in parsing folder"""
    # read train.json
    f = open("jointslu/temp/train.json", 'r')
    import json
    data = json.load(f)
    # remove bos and eos
    new_data = []
    for i in data:
        tok_len = len(i["text"].split(' '))
        new_text = ' '.join(i["text"].split(' ')[1:tok_len - 1])
        new_label = ' '.join(i["labels"].split(' ')[1:tok_len - 1])
        new_i = {
            "id": i["id"],
            "text": new_text,
            "labels": new_label
        }
        new_data.append(new_i)
    f.close()
    # store in file
    with open("jointslu/training_data/train.json", 'w') as f:
        json.dump(new_data,
                  f,
                  indent=4)
    f.close()


def get_labels_file_path(dataset, labels_version):
    """return path that stores original labels and simplified labels"""
    file_name = "labels" + labels_version + ".csv"
    return "data/" + dataset + "/labels/" + file_name


def get_simplified_labels(oris: List, ori_sim_dict):
    """ return a list of labels that matched from the original labels
    to simplified labels for one example"""
    labels = []
    for ori in oris:
        sim = "O"
        try:
            sim = ori_sim_dict[ori]
        except KeyError or IndexingError:
            print(f"[get_simplified_labels]\nread {ori} from dict error")
        labels.append(sim)
    return labels


def check_training_data(dataset, labels_version):
    """check whether data files are ready and return the paths,
    if not, construct new data according to dataset and labels_version"""
    folder_path = "data/" + dataset + "/training_data/labels" + labels_version
    print(f"[check_training_data]\nlv{labels_version}: training data should be stored in {folder_path}")
    train_p = folder_path + "/train.json"
    test_p = folder_path + "/test.json"
    val_p = folder_path + "/val.json"
    # if folder path exist and not empty, then the data exists and return True
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    if not (os.path.exists(train_p) and os.path.exists(test_p) and os.path.exists(val_p)):
        # construct training data
        for data_type in ["train", "test", "val"]:
            construct_training_data(dataset=dataset,
                                    data_type=data_type,
                                    labels_version=labels_version)
    return train_p, test_p, val_p


def construct_training_data(dataset, data_type, labels_version):
    """change the annotated data for training pre-trained model to simplified labels"""
    path = "data/" + dataset + "/training_data/" + data_type + ".json"
    print(f"[construct_training_data]\nconstructing {dataset}.{data_type} data from {path}")
    new_path = "data/" + dataset + "/training_data/labels" + labels_version + "/" \
               + data_type + ".json"
    file = open(path, 'r')
    ori_data = json.load(file)
    # new_data = construct_data(data, dataset, labels_version)
    new_data = simplify_labels_intent(ori_data, dataset, labels_version)
    with open(new_path, 'w+') as f:
        json.dump(new_data, f, indent=4)
        f.close()


def simplify_labels_intent(data, dataset, labels_version):
    """change all the labels in training data to defined simplified labels and intents,
    each label version matches a intent version."""
    # get ori-sim labels dict
    labels_dict = get_ori_sim_dict(dict_type="labels", dataset=dataset, labels_version=labels_version)
    intents_dict = get_ori_sim_dict(dict_type="intents", dataset=dataset, labels_version=labels_version)
    simplified_data = []
    for sample in data:
        sample["intent"] = intents_dict[sample["intent"]] if sample["intent"] in intents_dict.keys() else "unknown"
        sim_labels = []
        for ori_label in sample["labels"]:
            if ori_label == "[CLS]" or ori_label == "[SEP]":
                sim_labels.append(ori_label)
                continue
            if ori_label in labels_dict.keys():
                sim_labels.append(labels_dict[ori_label])
            else:
                sim_labels.append('O')
        sample["labels"] = sim_labels
        simplified_data.append(sample)
    return simplified_data


def construct_data(data, dataset, labels_version):
    """change the labels to simplified labels and add bos and eos to the label"""
    new_data = []
    i = 0
    # get ori-sim labels dict
    ori_sim_dict = get_ori_sim_dict(dataset=dataset, labels_version=labels_version)
    for ele in data:
        # labels list
        labels = ele["labels"].split(' ')
        new_labels = get_simplified_labels(labels[1: len(labels) - 1], ori_sim_dict)
        new_labels.insert(0, '[CLS]')
        new_labels.append('[SEP]')

        new_data.append({
            'id': i,
            'text': ele['text'].split(' '),
            'labels': new_labels
        })
        i = i + 1
    return new_data


def generate_gpt3_examples_file(dataset, labels_version, in_file):
    """select an example for each label and store into file, number suggests the labels_version"""
    examples = []
    label_dict = get_labels_dict(dataset=dataset, labels_version=labels_version)
    labels = list(label_dict.keys())
    # open pre-train training data file
    # todo: check whether training data exists, already checked???
    path = "data/" + dataset + "/training_data/labels" + labels_version + "/train.json"
    print(f"gpt3 {dataset}_lv{labels_version} example file does not exist, creating one from training data {path}")
    f = open(path, 'r')
    data = json.load(f)
    for idx, label in enumerate(labels):
        example = None
        for item in data:
            # if contains the target label, save this example
            if label in item["labels"]:
                example = {
                    "id": idx,
                    "intent": item["intent"],
                    "text": item["text"],
                    "labels": item["labels"]
                }
                break
        examples.append({
            "label": label,
            "example": example
        })
        print(f"example of {label} is\n{example}")
    f_in = open(in_file, 'w+')
    json.dump(examples, f_in, indent=4)
    f_in.close()
    f.close()
    print(f"created examples file under {in_file}")


def get_samples(file_path, model_name, num, do_shuffle):
    """return required data for each model,
    gpt3: return texts and labels without bos and eos labels
    parsing: return texts and labels without bos and eos
    pre-train: return texts and labels
    """
    f = open(file_path, 'r')
    import json
    from random import shuffle
    data = json.load(f)
    length = len(data) if num == 0 else num
    if do_shuffle:
        shuffle(data)
    intent = [i["intent"] for i in data[0:length]]

    if model_name == "gpt3" or model_name == "parsing":
        # remove BOS and EOS
        text = [i["text"][1: len(i["text"]) - 1] for i in data[0:length]]
        labels = [i["labels"][1: len(i["text"]) - 1] for i in data[0:length]]
        return text, labels, intent
    elif model_name == "pre-train":
        text = [i["text"] for i in data[0:length]]
        labels = [i["labels"] for i in data[0:length]]
        return text, labels, intent
