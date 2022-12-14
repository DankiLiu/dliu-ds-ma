"""
1. define input and output
2. load model
3. model output post-processing
4. evaluation functions
"""
from random import shuffle, randint
from typing import List
from util import read_keys_from_json
import pandas as pd


def get_std_gt(text: str, labels: List):
    """
    return standard ground truth of the given example, given input text and original labels (simplified).
    :text: (str) A input text string
    :labels: (list) A list of simplified labels
    :return standard ground truth of the given example
    """
    # example:
    # input_text:
    #   flights from pittsburgh to baltimore arriving between 4 and 5 pm
    # output_labels:
    #   from city: pittsburgh; to city: baltimore; arrive time: 4; arrive time end: 5 pm;
    text = text.split(' ')
    if not len(text) == len(labels):
        return None
    d = {}
    for i in range(len(text)):
        if labels[i] == 'O':
            continue
        if labels[i] in d.keys():
            d[labels[i]] = d[labels[i]] + ' ' + text[i]
        else:
            d[labels[i]] = text[i]
    output_labels = ''
    for k, v in d.items():
        output_labels = output_labels + str(k) + ': ' + str(v) + '; '
    return output_labels


def get_std_output_parsing(phrases: List, prediction: List):
    """
    return standard ground truth of the given example for Parsing method,
    given input text and original labels (simplified).
    :phrases: (list) A list of phrases generated from Parsing
    :prediction: (list) A list of simplified labels predicted by Parsing
    :return standard ground truth of the given example
    """
    assert len(phrases) == len(prediction)
    output = []
    for i, p in enumerate(phrases):
        if p != "":
            output.append(prediction[i] + ': ' + p)
    output = '; '.join(output)
    output = output.strip()
    return output


def std_gpt3_example(example):
    """return text: str and gt: list for an gpt3 example"""
    text = example["text"]
    labels = example["labels"]
    text = ' '.join(text)
    gt = get_std_gt(text, labels)
    return text, gt


def get_kv_pairs(num, model, random=True):
    """return random kv_pairs of prediction and ground truth for given example,
    process them into defined format, results are used to evaluate the model"""
    assert model in ["parsing", "gpt3", "pre-train"]
    output_path = 'data/jointslu/' + model + '/' + model + '_output.json'

    gt_key = "std_gt"
    p_key = "prediction" if model == "gpt3" else "std_output"

    values = read_keys_from_json(output_path, p_key, gt_key)
    predictions, std_gts = values[p_key], values[gt_key]
    num_example = num if num <= len(predictions) else len(predictions)
    print(f"get {num_example} examples")
    preds, gts = [], []
    if random:
        idxs = [i for i in range(len(predictions))]
        shuffle(idxs)
        for i in range(num_example):
            preds.append([predictions[idxs[i]]])
            gts.append([std_gts[idxs[i]]])

    preds, gts = predictions[:num_example], std_gts[:num_example]
    assert len(predictions) == len(std_gts)
    # prediction kv_pairs and std_gt kv_pairs for num examples
    p_gt_pairs = [(str2kv_pairs(preds[i]), str2kv_pairs(gts[i])) for i in range(num_example)]
    return p_gt_pairs


def str2kv_pairs(string):
    """given a prediction or std_gt (str), return a dictionary contains label-phrase pairs"""
    kvs = {}
    string = string.strip().replace('\n', '')
    kv_pairs = string.split(';')
    for kv in kv_pairs:
        if kv == '':
            continue
        pair = kv.split(':')
        try:
            if pair[0].strip() != '' or pair[1] != '':
                kvs[pair[0].strip()] = pair[1].strip()
                # print(f"key {pair[0].strip()} > value {kvs[pair[0].strip()]}")
        except IndexError:
            print(f"KeyValue Pair Index out of range. {pair}")
    return kvs


def merge_data(model_name, label_name):
    """merge scores entries with same model and label name"""
    file_path = "evaluation/jointslu_results/scores.csv"
    df = pd.read_csv(file_path)
    df_select = df.loc[(df['model_name'] == model_name) & (df['label_name'] == label_name)]
    print(df_select)
    if df_select.size == 0 or df_select.size == 1:
        return
    # remove original data in df
    df.drop(df[(df['model_name'] == model_name) & (df['label_name'] == label_name)].index)
    new_col = {
        'model_name': model_name,
        'label_name': label_name,
        'key_counter': df_select['key_counter'].sum(),
        'exp_counter': df_select['exp_counter'].sum(),
        'cor': df_select['cor'].mean(),
        'par': df_select['par'].mean(),
        'inc': df_select['inc'].mean(),
        'mis': df_select['mis'].mean(),
        'spu': df_select['spu'].mean(),
    }
    print("new column: ", new_col)


def load_labels_results():
    # load metrics results from jointslu_
    pass