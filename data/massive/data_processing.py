import json
from typing import Literal

from data.data_processing import labels_csv_to_json

DiaAnnotationFileDictKey = Literal[
    'id',
    'text',
    'intent',
    'labels'
]
OriDictKey = ["id", "locale", "partition", "scenario", "intent", "utt", "annot_utt", "worker_id"]
TrainDictKey = ["id", "scenario", "intent", "text", "labels"]


def check_labels_massive(labels_version, scenario):
    """check existence of intent and label files for massive dataset given labels_version
    create json files from csv files"""
    folder = "data/massive/labels/"
    intent_csv = folder + scenario + "_intents" + labels_version + ".csv"
    intent_json = folder + scenario + "_intents" + labels_version + ".json"
    labels_csv = folder + scenario + "_labels" + labels_version + ".csv"
    labels_json = folder + scenario + "_labels" + labels_version + ".json"
    import os
    if os.path.exists(intent_csv) and not os.path.exists(intent_json):
        labels_csv_to_json(intent_csv)
    if os.path.exists(labels_csv) and not os.path.exists(labels_json):
        labels_csv_to_json(labels_csv)


def generate_ori_with_scenario(scenario):
    """generate data with original labels given scenario, generate labels for that scenario"""
    train, test, val = training_data_from_ori(scenario)
    folder_name = "data/massive/training_data/"

    # get labels from train
    intents, labels = [], []
    for item in train:
        intents.append(item["intent"])
        labels = [*labels, *item["labels"][1:-1]]
    intents = list(set(intents))
    labels = list(set(labels))
    print(intents)
    # save intents and labels into file, labels_version = "00"
    import csv
    with open("data/massive/labels/" + scenario + "_intents00.csv", 'w+') as f:
        csv_writer = csv.writer(f)
        for intent in intents:
            csv_writer.writerow([intent])

    with open("data/massive/labels/" + scenario + "_labels00.csv", 'w+') as f:
        csv_writer = csv.writer(f)
        for label in labels:
            csv_writer.writerow([label])

    # save train data
    train_path = folder_name + scenario + "_train.json"
    train_f = open(train_path, 'w+')
    json.dump(train, train_f, indent=4)

    test_path = folder_name + scenario + "_test.json"
    test_f = open(test_path, 'w+')
    json.dump(test, test_f, indent=4)

    val_path = folder_name + scenario + "_val.json"
    val_f = open(val_path, 'w+')
    json.dump(val, val_f, indent=4)


def training_data_from_ori(scenario):
    """generate training data and save them in training_data folder given scenario
    name the folder"""
    # load data from ori_path
    ori_path = "data/massive/original_data/en-US.jsonl"

    ori_data = []
    with open(ori_path, 'r') as ori_file:
        for line in ori_file:
            ori_data.append(json.loads(line))
        ori_file.close()
    train, test, val = [], [], []
    # construct labels from annot_utt
    for i, row in enumerate(ori_data):
        if row["scenario"] != scenario:
            continue
        partition = row["partition"]
        intent = row["intent"]
        annot_utt = row["annot_utt"]
        text, labels = annot_utt_to_labels(annot_utt)
        data = {
            "id": i,
            "intent": intent,
            "text": text,
            "labels": labels
        }
        if partition == "train":
            train.append(data)
        elif partition == "test":
            test.append(data)
        elif partition == "dev":
            val.append(data)
    return train, test, val


def annot_utt_to_labels(annot_utt):
    text, labels = [], []
    ann_text, ann_label = False, False  # if both false, not in annotation range
    special_chars = ["[", "]", ":", " "]
    cur_text, cur_label = "", ""
    buffer = ""

    for i, char in enumerate(annot_utt):
        # add the char into buffer as long as it is not special character
        if char not in special_chars:
            buffer += char
            if i == len(annot_utt) - 1 and buffer != "":
                cur_text = buffer
                cur_label = 'O'
            continue
        if char == "[":
            ann_label = True
            ann_text = False
            continue
        if char == ":" and (ann_text or ann_label):
            ann_label = False
            ann_text = True
            continue
        if char == "]" and (ann_text or ann_label):
            if buffer != "":
                cur_text = buffer
                buffer = ""
            ann_text = False
            continue
        if char == " ":
            if buffer != "":
                if ann_label:
                    cur_label = buffer
                elif ann_text:
                    cur_text = buffer
                else:
                    cur_text = buffer
                    cur_label = 'O'
                buffer = ""
        if cur_text != "" and cur_label != "":
            text.append(cur_text)
            labels.append(cur_label)
            cur_text = ""
            if not (ann_text or ann_label):
                cur_label = 'O'
    if cur_text != "" and cur_label != "":
        text.append(cur_text)
        labels.append(cur_label)
    assert len(text) == len(labels)

    text.insert(0, "BOS")
    text.append("EOS")
    labels.insert(0, "[CLS]")
    labels.append("[SEP]")
    return text, labels
