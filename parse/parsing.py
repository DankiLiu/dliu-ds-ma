from typing import List

import pandas as pd

from data.data_processing import get_labels_dict
from evaluation.evaluation_utils import get_std_gt, get_std_output_parsing

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch

import matplotlib.pyplot as plt
import seaborn as sns
import json

# same labels as in pre-train
from parse.parsing_util import get_labels_ts_phrases
from util import append_to_json, get_parsing_params

f = open("data/jointslu/pre-train/labels.json")
data = json.load(f)
LABELS = list(data.keys())
f.close()


def plotting(labels=None, acc: List = None, f1: List = None):
    df = pd.read_csv("../data/jointslu/parsing/scores.csv")
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


def parse_testing(testing_file, num, model_version, dataset, output_file, labels_version):
    do_shuffle = get_parsing_params(model_version)
    parsing(testing_file, num, dataset, output_file, labels_version, do_shuffle)


def parsing(testing_file, num, dataset, output_file, labels_version, do_shuffle):
    results = []
    utexts, ulabels, parsed_phrases = get_labels_ts_phrases(testing_file=testing_file, num=num, do_shuffle=do_shuffle)
    assert len(utexts) == len(ulabels) == len(parsed_phrases) == num
    # load bert for similarity
    sbert = sbert_model()
    for n in range(num):
        # find label for each phrase with similarity
        # todo: load simplied labels by labels_version
        labels_dict = get_labels_dict(dataset=dataset,
                                      labels_version=labels_version)
        cosine_scores = cos_sim_per_example(parsed_phrases[n], labels_dict.keys(), sbert)
        label_idxs = [torch.argmax(i_score).item() for i_score in cosine_scores]
        prediction = [LABELS[idx] for idx in label_idxs]
        std_output = get_std_output_parsing(parsed_phrases[n], prediction)
        std_gt = get_std_gt(utexts[n], ulabels[n])
        # todo: construct ground truth
        result = {
            "num": n,
            "text": utexts[n],
            "gt": ulabels[n],
            "phrase": parsed_phrases[n],
            "prediction": prediction,
            "std_output": std_output,
            "std_gt": std_gt
        }
        print(f"result for {n}th example ", result)
        results.append(result)
    # todo: use modified append_to_json file
    append_to_json(file_path=output_file, new_data=results)
    print(f"{len(results)} results appended to parsing_output.json")


def evaluation(num=1, shuffle=True, pos=True, ner=True):
    """Evaluate parsing with all labels"""
    # read results from file and evaluate the parsing method
    pass


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
        print(cur_prediction)
        print(cur_labels)
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


