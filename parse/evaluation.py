from typing import List

import pandas as pd

import util
from data.data_processing import read_jointslu_labels_dict

from parse.nltk_parser import dependency_parsing
from parse.nltk_parser import name_entity_recognition as ner
from parse.nltk_parser import part_of_speech_parsing as pos

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch

import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv

# same labels as in pre-train
f = open("data/jointslu/pre-train/labels.json")
data = json.load(f)
LABELS = list(data.keys())
f.close()


def phrases_from_dep_graph(words, dep_graph):
    """Construct phrases from parsing result.
    (dependency parsing, POS and NER)"""
    phrases_dict = {}
    for i in range(len(words)):
        phrases_dict[i] = []
        # for loop start from 1 to sentence length
        dep_word = dep_graph.get_by_address(i + 1)
        deps = dep_word['deps']
        # todo: here check the pos of the word and apply some rules
        dep_idxs = [ele - 1 for sublist in deps.values() for ele in sublist]
        for idx in dep_idxs:
            phrase = words[idx] + ' ' + words[i]
            phrases_dict[i].append(phrase)
    return phrases_dict


def plotting(labels=None, acc: List = None, f1: List = None):
    df = pd.read_csv("../data/jointslu/parsing_eval/scores.csv")
    print(df.head())
    ndf = df.loc[df["num of examples"] == 1000]
    plt.ylim(0, 1.0)

    sns.barplot(x='num of examples',
                y='scores',
                data=ndf,
                palette='Paired',
                hue="type")

    plt.title('Accuracy and F1 scores of the parsing method')
    plt.show()


def main():
    sample_nums = [300, 1000]
    avg_accs = []
    avg_f1s = []
    for i in sample_nums:
        acc, f1 = evaluation(num=i, shuffle=True)
        avg_acc = sum(acc) / i
        avg_f1 = sum(f1) / i
        avg_accs.append(avg_acc)
        avg_f1s.append(avg_f1)
    print(avg_accs)
    print(avg_f1s)
    # todo: save evaluation scores into a .csv data


def evaluation_per_label(num, shuffle):
    """return acc and f1 for num of tests for each specific label"""
    # get training examples
    text, labels = get_examples("test", num=num + 20, do_shuffle=shuffle)
    gts = [ground_truth(label.split(' ')) for label in labels]
    predictions, _, utext, ulabels, ugts = model_testing(text, labels, gts)
    # todo: For each key (label), calculate how many times it is appeared in gt?
    #  or is it covered in tp and tn?
    f = open("data/jointslu/parsing_eval/par_eval_labels.csv", 'a')
    run_example = False
    if run_example:
        # test label "to city"
        TP, FN, FP, TN = prediction_metrics_per_label(predictions, ugts, "to city")
        # store result in csv file
        row = {
            "label_name": "to city",
            "num_examples": num,
            "TP": TP,
            "FN": FN,
            "FP": FP,
            "TN": TN
        }
        print(row)
    else:
        for label in LABELS:
            TP, FN, FP, TN = prediction_metrics_per_label(predictions, ugts, label)
            # store result in csv file
            row = {
                "label_name": label,
                "num_examples": num,
                "TP": TP,
                "FN": FN,
                "FP": FP,
                "TN": TN
            }
            print(row)
            # print(f"result for label {label} \n", row)
            # writer = csv.writer(f)
            # writer.writerow(row)
        f.close()


def model_testing(text, labels, gts):
    """return the result of parsing"""
    assert len(text) == len(labels)
    predictions, matching_dicts, utext, ulabels, ugts = [], [], [], [], []
    # parse examples
    dep_result, _, ner_result = parse_examples(text)
    # load sbert model
    sbert = sbert_model()
    ith_exp = 0
    while ith_exp < len(text):
        assert len(text[ith_exp].split(' ')) == len(gts[ith_exp])
        dp_graph = dep_result[ith_exp]
        ner_labels = ner_result[ith_exp]
        # compare dependency graph length with labels length
        min_add = len(gts[ith_exp]) - 3
        address = min_add
        while dp_graph.contains_address(min_add):
            address = min_add
            min_add = min_add + 1
        # if length not match, skip this example
        if address != len(gts[ith_exp]):
            print("length not match")
            ith_exp = ith_exp + 1
            continue
        phrases, pgt, matching_dict = interpret_dep(dp_graph, ner_labels, gts[ith_exp], with_ner=True)
        # store the used examples
        utext.append(text[ith_exp])
        ulabels.append(labels[ith_exp])
        ugts.append(gts[ith_exp])
        # find label for each phrase with similarity
        cosine_scores = cos_sim_per_example(phrases, LABELS, sbert)
        label_idxs = [torch.argmax(i_score).item() for i_score in cosine_scores]
        predicted = [LABELS[idx] for idx in label_idxs]
        predictions.append(predicted)
        matching_dicts.append(matching_dict)
        print("text: ", text[ith_exp])
        print("labels: ", labels[ith_exp])
        print("phrases: ", phrases)
        print("ground truth: ", gts[ith_exp])
        print("predicted: ", predicted)
    assert len(predictions) == len(utext)
    assert len(predictions) == len(ulabels)
    return predictions, matching_dicts, utext, ulabels, ugts


def evaluation(num=1, shuffle=True, pos=True, ner=True):
    """Evaluate parsing with all labels"""
    # get training examples
    text, labels = get_examples("test", num=num + 20, do_shuffle=shuffle)
    gts = [ground_truth(label.split(' ')) for label in labels]
    predictions, matching_dicts, utext, ulabels, ugts = model_testing(text, labels, gts)
    acc, f1 = [], []
    for i in range(len(predictions)):
        acc.append(accuracy_score(predictions[i], ugts[i], matching_dicts[i]))
        f1.append(f1_score(predictions[i], ugts[i], matching_dicts[i]))
    return acc, f1


def f1_score(prediction: List, labels: List, mapping: dict):
    tp, fn, fp, tn = prediction_metrics(prediction,
                                        labels,
                                        mapping)
    recall = get_recall(tp, fn, fp, tn)
    precision = get_precision(tp, fn, fp, tn)
    return F1(precision, recall)


def accuracy_score(prediction: List, labels: List, mapping: dict):
    tp, fn, fp, tn = prediction_metrics(
        prediction,
        labels,
        mapping)

    print(f"tp {tp}, fn {fn}, fp {fp}, tn {tn}")
    return accuracy(tp, fn, fp, tn)


def F1(precision, recall):
    if precision == 0 and recall == 0:
        return 0
    return 2 * (recall * precision) / (recall + precision)


def accuracy(tp, fn, fp, tn):
    return (tp + tn) / (tp + fp + fn + tn)


def get_precision(tp, fn, fp, tn):
    if tp == 0:
        return 0
    return tp / (tp + fp)


def get_recall(tp, fn, fp, tn):
    if tp == 0:
        return 0
    return tp / (tp + fn)


def prediction_metrics_per_label(predictions: List, labels: List, label):
    """return TP, FN, FP, TN of n selected label.
    preditions: predicted labels for n examples,
    labels: ground truth labels for n examples,
    label: the label to be evaluated"""
    TP, FN, FP, TN = 0, 0, 0, 0
    num = len(predictions)
    for i in range(num):
        cur_prediction, cur_labels = predictions[i], labels[i]
        # if the label is in labels, check if it is predicted
        if label in cur_labels:
            if label in cur_prediction:
                TP = TP + 1
            else:
                FN = FN + 1
        else:
            # if the label is not in labels, check if it is wrongly predicted
            if label in cur_prediction:
                FP = FP + 1
            else:
                TN = TN + 1
    return TP, FN, FP, TN


def prediction_metrics(prediction: List, labels: List, mapping: dict):
    """return TP, FN, FP, TN of the prediction (one prediction)
    prediction: a list of of labels for phrases
    labels: a list of simplified labels for current sentence (gt),
    mapping: a dict that maps phrase index to a list of words (idxs in
    original sentence)"""
    tp, fn, fp, tn = 0, 0, 0, 0
    keys, values = mapping.keys(), mapping.values()
    # idxs that are in the phrase and classified
    labeled_idx = list(set([item for sublist in values for item in sublist]))

    for i in range(len(prediction)):
        ''' for predicted words, check whether it is classified right '''
        if i in keys:
            # words in a phrase
            word_idxs = mapping.get(i)
            gt = []
            for idx in word_idxs:
                if labels[idx] != 'O':
                    gt.append(labels[idx])
            ''' if the phrase is classified right, then fn plus one,
            for each not classified label, fn plus one '''
            # todo: partial classification
            # todo: if this label is classified else where

            if prediction[i] != 'O' and prediction[i] in list(set(gt)):
                tp = tp + 1
                fn = fn + len(set(gt)) - 1
            else:
                fn = fn + len(set(gt))
        elif i not in labeled_idx:
            ''' word at this index are not classified,
            if the original label is 'O', the tn plus one,
            if the original label if not 'O', the fn plus one '''
            if labels[i] == 'O':
                tn = tn + 1
            else:
                fn = fn + 1
    return tp, fn, fp, tn


def ground_truth(labels):
    """return a list of labels as ground truth"""
    f = open("data/jointslu/labels.csv", 'r')
    import csv
    data = csv.reader(f)
    # generate labels-gt match with dict
    label2gt = {}
    for line in data:
        label2gt[line[0]] = line[1]
    gt = []
    for label in labels:
        try:
            gt.append(label2gt[label])
        except KeyError:
            gt.append('O')
    f.close()
    return gt


def parse_examples(text: List):
    """return the parsing result as a list"""
    # load nlp server
    if not util.server_is_running("http://localhost:9000/"):
        util.corenlp_server_start()
    dep_result = []
    pos_result = []
    ner_result = []
    for sentence in text:
        parsed_dep = dependency_parsing(sentence)
        parsed_pos = pos(sentence)
        parsed_ner = ner(sentence)
        dep_result.append(parsed_dep)
        pos_result.append(parsed_pos)
        ner_result.append(parsed_ner)
    return dep_result, pos_result, ner_result


def plot_sim_score(phrase, labels, cosine_scores):
    """plot cosine scores on 2D image"""
    x = torch.linspace(-5, 5, 100)
    x_squared = x * x

    plt.plot(x, x_squared)  # Fails: 'Tensor' object has no attribute 'ndim'

    torch.Tensor.ndim = property(lambda self: len(self.shape))  # Fix it

    plt.plot(x, x_squared)  # Works now
    plt.show()


def get_label_set():
    """return a set (list type) of labels from labels.csv file"""
    f = open("../data/jointslu/labels.csv", 'r')
    import csv
    data = csv.reader(f)
    labels = set([line[1] for line in data])
    print(f"{len(labels)} labels: {labels}")
    f.close()
    return list(labels)


def interpret_dep(dependency_graph, ner_labels, labels, with_ner=False):
    """return a list of phrases constructed from dependency graph"""
    phrases = []
    pgt = []
    address = 1
    # if current index has phrase, create a list of the words in the phrase
    # and store under the index key
    matching_dict = {}
    while True:
        word_dict = dependency_graph.get_by_address(address)
        if word_dict['address'] is None:
            break
        phrase = ""
        gt = ""
        # skip this word or not
        if not pos_skip(word_dict):
            # check dependency type, dep is a dep_dict
            idxs = []
            for key, value in word_dict['deps'].items():
                if not dep_skip(key):
                    # if not skip, construct phrase
                    for idx in value:
                        idxs.append(idx)
            idxs.append(address)
            idxs.sort()
            matching_dict[address - 1] = [i - 1 for i in idxs]
            # merge phrase labels
            gt = list(set([labels[i - 1] for i in idxs]))
            if len(gt) > 1 and 'O' in gt:
                gt.remove('O')
            phrase = construct_phrase(dependency_graph, idxs)
        elif labels[address - 1] != "":
            # if skipped word has labels, also reserve
            gt = labels[address - 1]
        if with_ner and phrase != "":
            if ner_labels[address - 1][1] != "O":
                phrase = phrase + ' ' + ner_labels[address - 1][1]
        pgt.append(gt)
        phrases.append(phrase)
        address = address + 1
    return phrases, pgt, matching_dict


def construct_phrase(dependency_graph, idxs):
    words = []
    for i in idxs:
        words.append(dependency_graph.get_by_address(i)['word'])
    return ' '.join(words)


def pos_skip(word_dict):
    """return True if the word has no dependency or labeled
    as VPB, otherwise False"""
    if len(word_dict["deps"]) == 0:
        return True
    if word_dict["ctag"] == "VBP" or word_dict["ctag"] == "VB":
        return True
    # if word_dict["ctag"] == "VB" -> goal action
    return False


def dep_skip(dep):
    skip_list = ["nsubj", "conj", "mark", "obl", "cc", "nmod", "iobj", "cop", "det"]
    if dep in skip_list:
        return True
    return False


def get_examples(data_type, num=0, do_shuffle=False):
    """get example from parsing_eval/train.json"""
    file_name = "data/jointslu/parsing_eval/train.json"
    if data_type == "test":
        file_name = "data/jointslu/parsing_eval/test.json"
    elif data_type == "val":
        file_name = "data/jointslu/parsing_eval/val.json"

    f = open(file_name, 'r')
    import json
    from random import shuffle
    # todo: not a real shuffle
    data = json.load(f)
    length = len(data) if num == 0 else num
    print("get ", length, " examples")
    if do_shuffle:
        shuffle(data)
    text = [i["text"] for i in data[0:length]]
    labels = [i["labels"] for i in data[0:length]]
    return text, labels


def cos_sim_per_example(phrases: List, labels: List, sbert):
    """
    return similarity dict of sentence phrases and labels.
    input: phrases: a list of phrases
           labels: a list of labels
    output: a matrix of cosine similarities
    """
    # encode labels
    label_embs = sbert.encode(labels, convert_to_tensor=True)
    phrase_embs = sbert.encode(phrases, convert_to_tensor=True)
    cosine_scores = cos_sim(phrase_embs, label_embs)
    return cosine_scores


def sbert_encoding(phrases: list):
    # create sentence bert model
    if not phrases:
        return None
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embs = model.encode(phrases)
    return embs


def sbert_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model


def accuracy_sim(predicted, gt):
    """return accuracy of the model in percentage """
    assert len(predicted) == len(gt)
    length = len(predicted)
    po_num = 0
    for i in range(length):
        if predicted[i] in gt[i] or predicted[i] == gt[i]:
            po_num = po_num + 1
    return po_num / length


def accuracy_gt(predicted, gt):
    """return accuracy of the model in percentage """
    assert len(predicted) == len(gt)
    length = len(predicted)
    po_num = 0
    for i in range(length):
        if predicted[i] == gt[i]:
            po_num = po_num + 1
    return po_num / length

