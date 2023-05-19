from evaluation.main import evaluate_bymodel, evaluate_intent, evaluate_slot
from main import test_model


def parsing_jointslu(num, model_version, labels_version, ger_num=-1):
    """Test parsing with jointslu, three times and calcualte """
    # test model and generate an output file
    gen = False
    if gen:
        test_model(model_name="parsing",
                   num=ger_num,
                   model_version=model_version,
                   dataset="jointslu",
                   labels_version=labels_version,
                   scenario=None,
                   few_shot_num=-1)

    # evaluate bymodel three times
    model_eva = True
    intent_eva = True
    slot_eva = True
    if model_eva:
        evaluate_bymodel(sample_num=num,
                         dataset="jointslu",
                         labels_version=labels_version,
                         generate=True,
                         scenario=None,
                         num_experiments=1,
                         model="parsing",
                         model_version=0)
    if slot_eva:
        evaluate_slot(sample_num=num,
                      dataset="jointslu",
                      labels_version=labels_version,
                      generate=True,
                      scenario=None,
                      num_experiments=1,
                      model="parsing",
                      model_version=0)
    if intent_eva:
        evaluate_intent(sample_num=num,
                        dataset="jointslu",
                        labels_version=labels_version,
                        generate=True,
                        scenario=None,
                        num_experiments=1,
                        model="parsing",
                        model_version=0)


def parsing_massive(num, model_version, labels_version, scenario, ger_num=-1):
    """Test parsing with jointslu, three times and calcualte """
    # test model and generate an output file
    gen = False
    if gen:
        test_model(model_name="parsing",
                   num=ger_num,
                   model_version=model_version,
                   dataset="massive",
                   labels_version=labels_version,
                   scenario=scenario,
                   few_shot_num=-1)

    # evaluate bymodel three times
    model_eva = True
    intent_eva = True
    slot_eva = True
    if model_eva:
        evaluate_bymodel(sample_num=num,
                         dataset="massive",
                         labels_version=labels_version,
                         generate=True,
                         scenario=scenario,
                         num_experiments=1,
                         model="parsing",
                         model_version=0)
    if slot_eva:
        evaluate_slot(sample_num=num,
                      dataset="massive",
                      labels_version=labels_version,
                      generate=True,
                      scenario=scenario,
                      num_experiments=1,
                      model="parsing",
                      model_version=0)
    if intent_eva:
        evaluate_intent(sample_num=num,
                        dataset="massive",
                        labels_version=labels_version,
                        generate=True,
                        scenario=scenario,
                        num_experiments=1,
                        model="parsing",
                        model_version=0)


def parsing_experiment1():
    """evaluate semantic parsing on jointslu with label_version 01 and 02 (zero-shot)"""
    parsing_jointslu(-1, 0, "01")
    parsing_jointslu(-1, 0, "02")


def parsing_experiment2():
    """evaluate semantic parsing on massive::scenario with label_version 00 (zero_shot)"""
    # scenarios = ["al"]
    for scenario in scenarios:
        parsing_massive(num=-1,
                        model_version=0,
                        labels_version="00",
                        scenario=scenario,
                        ger_num=-1)


if __name__ == '__main__':
    # parsing_experiment2()
    parsing_jointslu(-1, 0, "02", ger_num=-1)