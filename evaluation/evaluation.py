"""
evaluation functions for three model
the format and output of models should be the same for different models and different datasets
        result = {
            "num": (int) index of the example,
            "text": (List) a list of tokens (words) of user utterance,
            "intent_gt": (str) a phrase represents intent ground truth,
            "intent_prediction": (str) predicted intent,
            "gt": (List) a list of labels for each token,
            "prediction": (List) a list of predicted labels for each token,
            "phrase": [parsing approach] (List) a list of phrases constructed from parsing method,
            "std_output": (str) a string constructed from predictions into key-value pairs format,
            "std_gt": (str) a string constructed from ground truth into key-value pairs format
        }
            '''
        kv_pairs_dict[model_name] = {
            "gt_kv_pairs": gt_kv_pairs,
            "pd_kv_pairs": pd_kv_pairs
        }
    '''
"""
import csv, json
import pandas as pd

from data.data_processing import get_intents_labels_keys, get_intents_dict, get_labels_dict
from evaluation.evaluation_utils import process_data_to_kv_pairs
from evaluation.muc5 import MUC5
from util import get_output_folder

RESULTS_TYPES = ["equal", "under", "over", "mismatch", "mis", "spu", "non"]
MATCHING_TYPES = ["correct", "partial", "incorrect", "missing", "spurious"]


def model_evaluation(dataset, sample_num, mode, labels_version, scenario, model, num_experiment):
    """evaluate three approaches with num examples and store the result in results folder
    dataset: name of the dataset
    sample_num: number of samples to be evaluated
    bylabel: when true evaluate by label, otherwise false
    num: num of examples for each label"""
    # todo: some labels have very few occurrence, an option to evaluate all of the occurrence,
    #  see the result first, would not be difficult to add
    # load data (std_prediction and std_gt) from the most recent output files, raise exception if not enough examples
    print("load data dict")
    all_entries = []
    for i in range(num_experiment):
        loaded_data_dict = load_data_from_latest_model(dataset, sample_num, model)
        kv_pairs_dict = process_data_to_kv_pairs(loaded_data_dict)

        if mode == "intent":
            intents = get_intents_dict(dataset, labels_version, scenario).keys()
            i_entries = evaluate_models_by_intents(kv_pairs_dict, intents)
            all_entries = all_entries + i_entries
        elif mode == "slot":
            labels = get_labels_dict(dataset, labels_version, scenario).keys()
            s_entries = evaluate_models_by_labels(kv_pairs_dict, labels)
            all_entries = all_entries + s_entries
        else:
            # calculate metrics of all labels in the sample
            entries = evaluate_models(kv_pairs_dict)
            all_entries = all_entries + entries

    # save to file
    if mode == "intent":
        # save intent and slot to separate files
        intent_file = get_file_name("intent", dataset, labels_version, sample_num, model, num_experiment, scenario)
        f = open(intent_file, 'w+')
        # header = ["model_name", "label_name", "num", "correct", "partial", "incorrect", "missing", "spurious"]
        header = list(all_entries[0].keys())
        df = pd.DataFrame(all_entries, columns=header)
        df.to_csv(f)
        f.close()
    elif mode == "slot":
        slot_file = get_file_name("slot", dataset, labels_version, sample_num, model, num_experiment, scenario)
        f = open(slot_file, 'w+')
        # header = ["model_name", "label_name", "num", "correct", "partial", "incorrect", "missing", "spurious"]
        header = list(all_entries[0].keys())
        df = pd.DataFrame(all_entries, columns=header)
        df.to_csv(f)
        f.close()
    else:
        file = get_file_name("bymodel", dataset, labels_version, sample_num, model, num_experiment, scenario)
        f = open(file, 'w+')
        # header = ["model_name", "label_name", "num", "correct", "partial", "incorrect", "missing", "spurious"]
        header = list(all_entries[0].keys())
        df = pd.DataFrame(all_entries, columns=header)
        df.to_csv(f)
        f.close()


def get_file_name(type, dataset, labels_version, sample_num, model, num_experiment, scenario):
    file_index = 0
    scenario = "" if not scenario else scenario + "_"
    folder_name = "evaluation/" + model + "/" + dataset + "/" + type + "/" + scenario + "e" + str(num_experiment)
    file_name = "lv" + labels_version + "_n" + str(sample_num) + "_i" + str(file_index) + ".csv"
    import os
    while os.path.exists(folder_name + file_name):
        file_index += 1
        file_name = "lv" + labels_version + "_n" + str(sample_num) + "_i" + str(file_index) + ".csv"
    print("entries stored in: ", folder_name + file_name)
    return folder_name + file_name


def evaluate_models(kv_pairs_dict):
    """return metrics results of all three approaches evaluating all given label
    :kv_pairs_dict: dictionary that stores model_name and its gt_kv_pairs and pd_kv_pairs
    :labels: labels to be evaluated
    :num: num of labels to be evaluated"""
    entries = []
    for model_name, kv_pairs in kv_pairs_dict.items():
        # for each model, each example, check the matching type of each prediction
        gt_kv_pairs = kv_pairs["gt_kv_pairs"]
        pd_kv_pairs = kv_pairs["pd_kv_pairs"]

        metrics = get_metrics_model(gt_kv_pairs, pd_kv_pairs)
        metrics_dict = dict(zip(MATCHING_TYPES, metrics))
        entry = {
            "model_name": model_name,
            "num": len(gt_kv_pairs)
        }
        entry.update(metrics_dict)
        entries.append(entry)
        print("[evaluate all samples] entries keys", entry.keys())
    return entries


def get_metrics_model(gt_kv_pairs, pd_kv_pairs):
    """:return metrics, a list of metrics count the matching types of all samples in parsing or pretrain approach
    parsing and pretrain contain only the defined labels"""
    metrics = [0, 0, 0, 0, 0]
    assert len(gt_kv_pairs) == len(pd_kv_pairs)
    for gt_kv_pair, pd_kv_pair in zip(gt_kv_pairs, pd_kv_pairs):
        # (gt_kv_pair, pd_kv_pair) is a dict of label-phrase for one sample
        example_matching_types = get_example_matching_type(gt_kv_pair, pd_kv_pair)
        for matching_type_index in example_matching_types:
            metrics[matching_type_index] += 1
    return metrics


def get_example_matching_type(gt_kv_pair, pd_kv_pair):
    # get keys set of gt and prediction
    keys = list(set(list(gt_kv_pair.keys()) + list(pd_kv_pair.keys())))
    example_matching_types = []
    for key in keys:
        example_matching_types.append(get_matching_type(gt_kv_pair, pd_kv_pair, key))
    return example_matching_types


def evaluate_models_by_intents(kv_pairs_dict, intents):
    entries = []
    # check label num occurrences
    for model_name, kv_pairs in kv_pairs_dict.items():
        print(f"[evaluate_models_by_labels] model_name: {model_name}")
        for intent in intents:
            print(f"evaluate intent - {intent}")
            num_evaluated, sample_index, metrics = get_metrics_intent(gt_kv_pairs=kv_pairs["gt_kv_pairs"],
                                                                      pd_kv_pairs=kv_pairs["pd_kv_pairs"],
                                                                      intent_name=intent)
            # print(f"[evaluate_models_by_labels] get metrics {metrics}")
            metrics_dict = dict(zip(MATCHING_TYPES, metrics))
            entry = {
                "model_name": model_name,
                "label_name": intent,
                "sample_num": sample_index,
                "num": num_evaluated
            }
            entry.update(metrics_dict)
            entries.append(entry)
            # print("entries keys", entry.keys())
    return entries


def evaluate_models_by_labels(kv_pairs_dict, labels):
    """return metrics results of all three approaches evaluating all given label,
    if occurrences of label is less than num, require new num
    :kv_pairs_dict: dictionary that stores model_name and its gt_kv_pairs and pd_kv_pairs
    :labels: labels to be evaluated
    :num: num of labels to be evaluated"""
    entries = []
    # check label num occurrences
    for model_name, kv_pairs in kv_pairs_dict.items():
        print(f"[evaluate_models_by_labels] model_name: {model_name}")
        for label in labels:
            num_evaluated, sample_index, metrics = get_metrics_label(gt_kv_pairs=kv_pairs["gt_kv_pairs"],
                                                                     pd_kv_pairs=kv_pairs["pd_kv_pairs"],
                                                                     label_name=label)
            # print(f"[evaluate_models_by_labels] get metrics {metrics}")
            metrics_dict = dict(zip(MATCHING_TYPES, metrics))
            entry = {
                "model_name": model_name,
                "label_name": label,
                "sample_num": sample_index,
                "num": num_evaluated
            }
            entry.update(metrics_dict)
            entries.append(entry)
            # print("entries keys", entry.keys())
    return entries


def get_metrics_intent(gt_kv_pairs, pd_kv_pairs, intent_name):
    metrics = [0, 0, 0, 0, 0]
    # if label exists in the example, then evaluate, util num is reached
    num_evaluated = 0
    sample_index = 0
    for gt_kv_pair, pd_kv_pair in zip(gt_kv_pairs,
                                      pd_kv_pairs):  # if evaluated samples number is less than num, continue to evaluate
        if not contains_intent(gt_kv_pair, pd_kv_pair, intent_name):
            print(f"[get metrics label]: {intent_name} not in gt and pd")
            sample_index += 1
            continue
        else:
            print(f"[get metrics label]: get {intent_name} metrics")
            matching_type_index = get_matching_type_intent(gt_kv_pair=gt_kv_pair,
                                                           pd_kv_pair=pd_kv_pair,
                                                           intent=intent_name)
            metrics[matching_type_index] += 1
            num_evaluated += 1
            sample_index += 1
    return num_evaluated, sample_index, metrics


def get_metrics_label(gt_kv_pairs, pd_kv_pairs, label_name):
    """:return metrics, a list of scores evaluated label, metrics counts the matching types of num of samples contains
    the label_name"""
    metrics = [0, 0, 0, 0, 0]
    # if label exists in the example, then evaluate, util num is reached
    num_evaluated = 0
    sample_index = 0
    for gt_kv_pair, pd_kv_pair in zip(gt_kv_pairs,
                                      pd_kv_pairs):  # if evaluated samples number is less than num, continue to evaluate

        if not contains_label(gt_kv_pair, pd_kv_pair, label_name):
            # print(f"[get metrics intent]: {label_name} not in gt and pd")
            sample_index += 1
            continue
        else:
            print(f"[get metrics label]: get {label_name} metrics")
            matching_type_index = get_matching_type(gt_kv_pair=gt_kv_pair,
                                                    pd_kv_pair=pd_kv_pair,
                                                    label=label_name)

            metrics[matching_type_index] += 1
            num_evaluated += 1
            sample_index += 1
    return num_evaluated, sample_index, metrics


def load_data_from_latest(dataset, num):
    model_names = ["parsing", "pre-train", "gpt3"]
    latest_files = [find_latest_output_files(model, dataset) for model in model_names]
    print(f"[output file]: {latest_files} ")
    # check each file for num of examples, if less than given num, raise error
    loaded = []
    for i, model_name in enumerate(model_names):
        # load data, if num of examples are less than num, raise error
        with open(latest_files[i], 'r') as f:
            data = json.load(f)
            if len(data) < num:
                raise Exception(f"number of examples in {model_names[i]} output file is {len(data)}, less than {num}")
            else:
                from random import shuffle
                shuffle(data)
                gt_key = "std_gt"
                prediction_key = "prediction" if model_name == "gpt3" else "std_output"
                random_loaded = {
                    "std_gts": [item[gt_key] for item in data[:num]],
                    "predictions": [item[prediction_key] for item in data[:num]]
                }
                loaded.append(random_loaded)
        f.close()
    return dict(zip(model_names, loaded))


def load_data_from_latest_model(dataset, num, model):
    latest_files = find_latest_output_files(model, dataset)
    print(f"[output file for {model}]: {latest_files} ")
    # check each file for num of examples, if less than given num, raise error
    loaded = {}
    with open(latest_files, 'r') as f:
        data = json.load(f)
        num = len(data) if num < 0 else num
        if len(data) < num:
            raise Exception(f"number of examples in {model} output file is {len(data)}, less than {num}")
        else:
            from random import shuffle
            shuffle(data)
            gt_key = "std_gt"
            prediction_key = "prediction" if model == "gpt3" else "std_output"
            loaded = {
                "std_gts": [item[gt_key] for item in data[:num]],
                "predictions": [item[prediction_key] for item in data[:num]]
            }
        f.close()
    return {model: loaded}


def find_latest_output_files(model, dataset):
    """load the most recent output file from each approach"""
    # todo: this is a temporary solution (how to find output file?)
    output_folder = get_output_folder(model, dataset)
    import glob
    import os
    list_of_files = glob.glob(output_folder + '*.json')
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def acc_f1_by_model(model_name, is_loose, bymodel_path):
    """calculate acc and f1 of all given models for all given models, the metrics path is given"""
    # find entries in file that contain model_name, should be only one entry in one file
    df = pd.read_csv(bymodel_path)
    model_df = df.loc[(df["model_name"] == model_name)]
    # read metrics from file
    # calculate acc and f1 given is_loose
    # return the acc and f1
    # todo: what to do with the num column
    pass


def evaluate_acc_f1(model_name, label_name, is_loose, file_path):
    """
    dataframe of scores.csv: model_name,label_name,key_counter,exp_counter,cor,par,inc,mis,spu
    :param model_name: str, model name ["gtp3", "pre-train", "parsing"]
    :param label_name: str, simplified label name
    :return: None, store result in the file
    """
    result = None
    # read file from bylabel or bymodel file
    df = pd.read_csv(file_path)
    # separate into two functions, by label or by model
    print(df.head())
    accs, f1s = [], []
    num = 0
    # when by label, choose entries with model_name and label_name, should be only one row in a file
    model_df = df.loc[(df["model_name"] == model_name) & (df["label_name"] == label_name)]
    # calculate f1 and acc for each example
    for index, row in model_df.iterrows():
        print("num: ", num)
        print(f"{index} row : {row}")
        counter = row["exp_counter"] if row["label_name"] == "ALL" else row["key_counter"]
        if counter == 0:
            # counter == 0 means there is no example available for given model_name and label_name
            # todo: counter == 0 example should not be stored in the first place, check evaluate_model function
            continue
        cor, par, inc, mis, spu = row["cor"] / counter, row["par"] / counter, \
                                  row["inc"] / counter, row["mis"] / counter, row["spu"] / counter
        muc = MUC5(cor, par, inc, mis, spu)
        acc, f1 = muc.acc, muc.f
        accs.append(acc)
        f1s.append(f1)
    if len(accs) == len(f1s) == 0:
        # no entry found
        return
    try:
        print(f"acc [{model_name}, {label_name}]: ", accs)
        print(f"f1 [{model_name}, {label_name}]: ", f1s)
        acc, f1 = sum(accs) / len(accs), sum(f1s) / len(f1s)
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


def get_matching_type_intent(gt_kv_pair, pd_kv_pair, intent):
    """
    :return integer that represents a matching type of given example (given gt and pd kv pairs)
    matching types: correct, partial, incorrect, missing, spurious = 0, 1, 2, 3, 4
    comparison types: equal, under, over, mismatch, mis, spu, non = 0, 1, 2, 3, 4, 5, 6
    """
    keys = list(set(list(pd_kv_pair.keys()) + list(gt_kv_pair.keys())))  # merge keys, get a list of all keys
    # get both intent, and compare the value -> cor, par, inc
    # p intent is empty mis
    gt_intent = gt_kv_pair["intent"]
    p_intent = ""
    try:
        p_intent = pd_kv_pair["intent"]
    except:
        print("[get_matching_type_intent] no p_intent in prediction")
    if intent in [gt_intent, p_intent]:
        result_of_comparison = 0
        if gt_intent and p_intent:  # if both contain the label, compare
            # todo: function to compare p and gt values
            result_of_comparison = compare_values(gt_value=gt_intent, p_value=p_intent)
        elif gt_intent and not p_intent:
            # if gt has the label but prediction does not, it is a miss
            result_of_comparison = RESULTS_TYPES.index("mis")
        elif p_intent and not gt_intent:
            # if the label is predicted but it is not in gt, then it is a spu
            result_of_comparison = RESULTS_TYPES.index("spu")
        # return matching_type index w.r.t comparison result
        if result_of_comparison == 0:
            return MATCHING_TYPES.index("correct")
        elif result_of_comparison == 1 or result_of_comparison == 2:
            return MATCHING_TYPES.index("partial")
        elif result_of_comparison == 3:
            return MATCHING_TYPES.index("incorrect")
        elif result_of_comparison == 4:
            return MATCHING_TYPES.index("missing")
        elif result_of_comparison == 5:
            return MATCHING_TYPES.index("spurious")


def get_matching_type(gt_kv_pair, pd_kv_pair, label):
    """
    :return integer that represents a matching type of given example (given gt and pd kv pairs)
    matching types: correct, partial, incorrect, missing, spurious = 0, 1, 2, 3, 4
    comparison types: equal, under, over, mismatch, mis, spu, non = 0, 1, 2, 3, 4, 5, 6
    """

    keys = list(set(list(pd_kv_pair.keys()) + list(gt_kv_pair.keys())))  # merge keys, get a list of all keys
    # if both contains key, compare the value -> cor, par, inc
    # if gt contains key, p not, mis
    # if p contains key, gt not, spu
    for key in keys:
        if label is not None:
            if key != label:
                # for evaluating specific label, only exam when key == label
                continue
        result_of_comparison = 0
        if key in pd_kv_pair.keys() and key in gt_kv_pair.keys():  # if both contain the label, compare
            # todo: function to compare p and gt values
            gt_value, p_value = gt_kv_pair[key], pd_kv_pair[key]
            result_of_comparison = compare_values(gt_value=gt_value, p_value=p_value)
        elif key in gt_kv_pair.keys() and not key in pd_kv_pair.keys():
            # if gt has the label but prediction does not, it is a miss
            result_of_comparison = RESULTS_TYPES.index("mis")
        elif key in pd_kv_pair.keys() and not key in gt_kv_pair.keys():
            # if the label is predicted but it is not in gt, then it is a spu
            result_of_comparison = RESULTS_TYPES.index("spu")
        # return matching_type index w.r.t comparison result
        if result_of_comparison == 0:
            return MATCHING_TYPES.index("correct")
        elif result_of_comparison == 1 or result_of_comparison == 2:
            return MATCHING_TYPES.index("partial")
        elif result_of_comparison == 3:
            return MATCHING_TYPES.index("incorrect")
        elif result_of_comparison == 4:
            return MATCHING_TYPES.index("missing")
        elif result_of_comparison == 5:
            return MATCHING_TYPES.index("spurious")


def compare_values(p_value, gt_value):
    """
    compare the values and return their matching type
    equal, under, over, mismatch, mis, spu, non = 0, 1, 2, 3, 4, 5, 6
    """
    # todo: contain is not complete
    if p_value != '' and gt_value != '':
        if p_value == gt_value:
            return RESULTS_TYPES.index("equal")
        if p_value in gt_value:
            return RESULTS_TYPES.index("under")
        if gt_value in p_value:
            return RESULTS_TYPES.index("over")
        if p_value != gt_value:
            return RESULTS_TYPES.index("mismatch")
    else:
        if p_value == '' and gt_value != '':
            return RESULTS_TYPES.index("mis")
        elif gt_value == '' and p_value != '':
            return RESULTS_TYPES.index("spu")
        else:
            return RESULTS_TYPES.index("non")


def find_largest_num():
    """find the largest number of examples to evaluate"""
    pass


def label_num_occurrences(gt_kv_pairs, pd_kv_pairs, label):
    """:return number of occurrences of the given model in gt and kv pairs"""
    num = 0
    for gt_kv_pair, pd_kv_pair in zip(gt_kv_pairs, pd_kv_pairs):
        if contains_label(gt_kv_pair, pd_kv_pair, label):
            num += 1
    return num


def contains_label(gt_kv_pair, pd_kv_pair, label):
    """if label is defined, return whether given example contains label, if not defined, return true"""
    # todo: why return true if not defined? is that correct?
    if label is None:
        return True
    else:
        keys = list(set(list(gt_kv_pair.keys()) + list(pd_kv_pair.keys())))
        return label in keys


def contains_intent(gt_kv_pair, pd_kv_pair, intent):
    """if label is defined, return whether given example contains label, if not defined, return true"""
    # todo: why return true if not defined? is that correct?
    if intent is None:
        return True
    else:
        try:
            keys = [gt_kv_pair["intent"], pd_kv_pair["intent"]]
            return intent in keys
        except:
            print("can not read intent from pd_kv_pairs")
            return intent in [gt_kv_pair["intent"]]