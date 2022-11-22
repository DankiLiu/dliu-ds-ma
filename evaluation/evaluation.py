"""
evaluation functions for three model
the format and output of models should be the same for different models and different datasets
"""

from evaluation.evaluation_utils import get_kv_pairs

# matching_type = enumerate(["equal", "under", "over", "mismatch", "mis", "spu"])


def muc_recall(cor, par, inc, mis, spu):
    pass


def muc_precision(cor, par, inc, mis, spu):
    pass


def muc_f(recall, precision, b):
    pass


def muc_5_metrics(num, model, label=None):
    # get num of model outputs, iterate the outputs, get metrics for each output and add them together
    kv_pairs = get_kv_pairs(num=num, model=model)
    print("kv pairs: ", kv_pairs)
    matching_dict = {
        "equal": 0,
        "under": 0,
        "over": 0,
        "mismatch": 0,
        "mis": 0,
        "spu": 0
    }
    for i, pair in enumerate(kv_pairs):
        print(f"$ {i}th example: ")
        muc_metrics_per_example(pair, matching_dict, label)
        print(matching_dict)
    cor = matching_dict["equal"]
    par = matching_dict["under"] + matching_dict["over"]
    inc = matching_dict["mismatch"]
    mis = matching_dict["mis"]
    spu = matching_dict["spu"]
    return cor, par, inc, mis, spu


def muc_metrics_per_example(p_gt_pair, matching_dict, label=None):
    """calculate metrics for given example
    COR: p == gt
    PAR: p := gt
    INC: p != gt
    MIS: gt has key, p does not
    SPU: gt does not has key, p has
    NON: both do not have predictions
    """
    p_dict, gt_dict = p_gt_pair
    print("prediction dict: ", p_dict)
    print("ground truth dict: ", gt_dict)
    keys = list(set(list(p_dict.keys()) + list(gt_dict.keys())))
    # 1. merge all keys,
    # if both contains key, compare the value, cor, par, inc
    # if gt contains key, p not, mis
    # if p contains key, gt not, spu
    for key in keys:
        if label is not None:
            if key != label:
                # for evaluating specific label, only exam when key == label
                continue
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
            print(f"-- key <{key}> is <{matching_type}>, {matching_dict[matching_type]} + 1")


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
