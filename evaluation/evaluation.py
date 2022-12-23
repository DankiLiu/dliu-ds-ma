"""
evaluation functions for three model
the format and output of models should be the same for different models and different datasets
"""
import csv
import pandas as pd
from evaluation.evaluation_utils import get_kv_pairs


# matching_type = enumerate(["equal", "under", "over", "mismatch", "mis", "spu"])
def evaluate_acc_f1(model_name, label_name, is_loose):
    """
    dataframe of scores.csv: model_name,label_name,key_counter,exp_counter,cor,par,inc,mis,spu
    :param model_name: str, model name ["gtp3", "pre-train", "parsing"]
    :param label_name: str, simplified label name
    :return: None, store result in the file
    """
    result = None
    df = pd.read_csv("evaluation/jointslu_results/scores.csv")
    # print(df.head())
    model_df = df.loc[(df["model_name"] == model_name) & (df["label_name"] == label_name)]
    print(model_df)
    accs, f1s = [], []
    num = 0
    for index, row in model_df.iterrows():
        print("num: ", num)
        print(f"{index} row : {row}")
        counter = row["exp_counter"] if row["label_name"] == "ALL" else row["key_counter"]
        if counter == 0:
            # counter == 0 means there is no example available for given model_name and label_name
            # todo: counter == 0 example should not be stored in the first place, check evaluate_model function
            continue
        cor, par, inc, mis, spu = row["cor"]/counter, row["par"]/counter, \
                                  row["inc"]/counter, row["mis"]/counter, row["spu"]/counter
        num = num + counter
        acc, f1 = muc_loose_evaluation(cor, par, inc, mis, spu) if is_loose else \
            muc_strict_evaluation(cor, par, inc, mis, spu)
        accs.append(acc)
        f1s.append(f1)
    if len(accs) == len(f1s) == 0:
        # no entry found
        return
    try:
        print(f"acc [{model_name}, {label_name}]: ", accs)
        print(f"f1 [{model_name}, {label_name}]: ", f1s)
        acc, f1 = sum(accs)/len(accs), sum(f1s)/len(f1s)
        # todo: if contains model_name, label_name, is_loose entry, then merge those entry
        result = [model_name, label_name, num, is_loose, acc, f1]
        print("result: ", result)
    except ZeroDivisionError:
        print(f"divsion by zero - {len(accs)} acc, {len(f1s)} f1")
    # save reslut in file
    if result is not None:
        if False:
            with open("evaluation/jointslu_results/acc_f1.csv", 'a') as f:
                writer = csv.writer(f)
                writer.writerow(result)
                f.close()
    else:
        print(f"no entry met [{model_name}, {label_name}]")


def muc_loose_evaluation(cor, par, inc, mis, spu):
    # count par as correct
    cor = cor + par
    acc = muc_accuracy(cor, 0, inc, mis, spu)
    f1 = muc_f(cor, 0, inc, mis, spu)
    return acc, f1


def muc_strict_evaluation(cor, par, inc, mis, spu):
    # separate partial with right classification
    acc = muc_accuracy(cor, par, inc, mis, spu)
    f1 = muc_f(cor, par, inc, mis, spu)
    return acc, f1


def muc_accuracy(cor, par, inc, mis, spu):
    total = sum([cor, par, inc, mis, spu])
    if total == 0:
        return 0
    return cor/total


def muc_possible(cor, par, inc, mis, spu):
    return cor + par + inc + mis


def muc_actual(cor, par, inc, mis, spu):
    return cor + par + inc + spu


def muc_recall(cor, par, inc, mis, spu):
    denominator = cor + (par * 0.5)
    numerator = float(muc_actual(cor, par, inc, mis, spu))
    if numerator == 0:
        return 0
    return denominator/numerator


def muc_precision(cor, par, inc, mis, spu):
    denominator = cor + (par * 0.5)
    numerator = float(muc_possible(cor, par, inc, mis, spu))
    if numerator == 0:
        return 0
    return denominator/numerator


def muc_f(cor, par, inc, mis, spu, b=1.0):
    recall = muc_recall(cor, par, inc, mis, spu)
    precision = muc_precision(cor, par, inc, mis, spu)
    denominator = (b * b + 1.0) * precision * recall
    numerator = (b * b * precision) + recall
    if numerator == 0:
        return 0
    return denominator/numerator


def evaluate_model(model_name, num_exps, label_name=None):
    kv_pairs = get_kv_pairs(num=num_exps, model=model_name)
    # exp_counter counts how many examples are evaluated
    # key_counter counts how many time is key=label_name evaluated
    exp_counter = 0
    key_counter = 0
    correct, partial, incorrect, missing, spurious = 0, 0, 0, 0, 0
    for pair in kv_pairs:
        # if label_name is None, counter should be negative
        matching_dict, counter = get_muc_metrics_dict(p_gt_pair=pair, label=label_name)
        if counter > 0:
            key_counter = key_counter + counter
        exp_counter = exp_counter + 1
        cor, par, inc, mis, spu = get_5_metrics(matching_dict)
        correct, partial, incorrect, missing, spurious = \
            correct + cor, partial + par, incorrect + inc, missing + mis, spurious + spu
    # store result into file
    label = label_name if label_name is not None else "ALL"
    result = [model_name, label, key_counter, exp_counter, correct, partial, incorrect, missing, spurious]
    file_path = "evaluation/jointslu_results/scores.csv"
    with open(file_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(result)
        f.close()


def contains_label(p_gt_pair, label):
    """if label is defined, return whether given example contains label, if not defined, return truth"""
    if label is None:
        return True
    else:
        p_dict, gt_dict = p_gt_pair
        keys = list(set(list(p_dict.keys()) + list(gt_dict.keys())))
        return label in keys


def get_5_metrics(matching_dict):
    # count partial as right classification
    cor = matching_dict["equal"]
    par = matching_dict["under"] + matching_dict["over"]
    inc = matching_dict["mismatch"]
    mis = matching_dict["mis"]
    spu = matching_dict["spu"]
    return cor, par, inc, mis, spu


def get_muc_metrics_dict(p_gt_pair, label=None):
    """return a dictionary of comparison of each keys given an example,
    return number of evaluated keys, return positive number if label is not None, else -1
        "equal": both keys have same value,
        "under": predicted key has part of gt value,
        "over": predicted key has more than gt value,
        "mismatch": predicted value does not match gt value,
        "mis": no predictions but has gt,
        "spu": no gt but has prediction
    """
    matching_dict = {
        "equal": 0,
        "under": 0,
        "over": 0,
        "mismatch": 0,
        "mis": 0,
        "spu": 0
    }
    p_dict, gt_dict = p_gt_pair
    keys = list(set(list(p_dict.keys()) + list(gt_dict.keys())))
    # key_counter counts how many keys are given label
    key_counter = 0 if label is not None else -1
    # 1. merge all keys,
    # if both contains key, compare the value, cor, par, inc
    # if gt contains key, p not, mis
    # if p contains key, gt not, spu
    for key in keys:
        if label is not None:
            if key != label:
                # for evaluating specific label, only exam when key == label
                continue
            else:
                key_counter = key_counter + 1
        matching_type = ""
        if key in p_dict and key in gt_dict:
            # if both prediction and gt have the label, compare two
            # todo: function to compare p and gt values
            gt_value, p_value = p_dict[key], gt_dict[key]
            matching_type = get_matching_type(p_value=p_value, gt_value=gt_value)
        elif key in gt_dict and not key in p_dict:
            # if gt has the label but prediction does not, it is a miss
            matching_type = "mis"
        elif key in p_dict and not key in gt_dict:
            # if the label is predicted but it is not in gt, then it is a spu
            matching_type = "spu"
        if matching_type in matching_dict:
            matching_dict[matching_type] = matching_dict[matching_type] + 1
            # print(f"-- key <{key}> is <{matching_type}>, {matching_dict[matching_type]} + 1")
    return matching_dict, key_counter


def get_matching_type(p_value, gt_value):
    """compare the values and return their matching type"""
    # todo: what about empty value?
    # todo: contain is not complete
    if p_value != '' and gt_value != '':
        if p_value == gt_value:
            return "equal"
        if p_value in gt_value:
            return "under"
        if gt_value in p_value:
            return "over"
        if p_value != gt_value:
            return "mismatch"
    else:
        if p_value == '' and gt_value != '':
            return "mis"
        elif gt_value == '' and p_value != '':
            return "spu"
        else:
            return "non"
