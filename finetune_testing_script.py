from evaluation.main import evaluate_bymodel, evaluate_slot, evaluate_intent
from main import test_model


def pretrain_jointslu(num, model_version, labels_version, ger_num=-1):
    """Test parsing with jointslu, three times and calcualte """
    # test model and generate an output file
    gen = True
    if gen:
        test_model(model_name="pre-train",
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
                         model="pre-train",
                         model_version=model_version)
    if slot_eva:
        evaluate_slot(sample_num=num,
                      dataset="jointslu",
                      labels_version=labels_version,
                      generate=True,
                      scenario=None,
                      num_experiments=1,
                      model="pre-train",
                      model_version=model_version)
    if intent_eva:
        evaluate_intent(sample_num=num,
                        dataset="jointslu",
                        labels_version=labels_version,
                        generate=True,
                        scenario=None,
                        num_experiments=1,
                        model="pre-train",
                        model_version=model_version)


def pretrain_massive(num, model_version, labels_version, scenario, ger_num=-1):
    """Test parsing with jointslu, three times and calcualte """
    # test model and generate an output file
    gen = True
    if gen:
        test_model(model_name="pre-train",
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
                         model="pre-train",
                         model_version=model_version)
    if slot_eva:
        evaluate_slot(sample_num=num,
                      dataset="massive",
                      labels_version=labels_version,
                      generate=True,
                      scenario=scenario,
                      num_experiments=1,
                      model="pre-train",
                      model_version=model_version)
    if intent_eva:
        evaluate_intent(sample_num=num,
                        dataset="massive",
                        labels_version=labels_version,
                        generate=True,
                        scenario=scenario,
                        num_experiments=1,
                        model="pre-train",
                        model_version=model_version)


def pretrain_experiment1():
    """test pretrain model 2.0 on ATIS-lv02"""
    pretrain_jointslu(num=-1,
                      model_version=2.0,
                      labels_version="02",
                      ger_num=-1)


def pretrain_experiment2():
    """test pretrain model 2.0 finetuned on massive::scenario """
    scenario_dict = {"alarm": 2.1,
                     "audio": 2.2,
                     "iot": 2.3,
                     "music": 2.4,
                     "news": 2.5,
                     "takeaway": 2.6,
                     "weather": 2.7}
    for key, value in scenario_dict.items():
        pretrain_massive(num=-1,
                         model_version=value,
                         labels_version="00",
                         scenario=key,
                         ger_num=-1)


def pretrain_experiment3():
    """test pretrain model 2.0 fintuned on massive::alarm under fewshot setting"""
    few_shot_number = [200]
    for num in few_shot_number:
        pretrain_massive(num=num,
                         model_version=2.0,
                         labels_version="00",
                         scenario="alarm",
                         ger_num=-1)


def pretrain_experiment4():
    """test pretrain model 2.0 fintuned on massive::scenarios on all training
    max_epoch=300 with early stopping"""
    scenario_dict = {"alarm": 3.1,
                     "audio": 3.2,
                     "iot": 3.3}
    for scenario, model_version in scenario_dict.items():
        pretrain_massive(num=-1,
                         model_version=model_version,
                         labels_version="00",
                         scenario=scenario,
                         ger_num=-1)


def pretrain_experiment5():
    """test pretrain model 2.0 fintuned on massive::scenarios under fewshot setting
    max_epoch=300 with early stopping"""
    scenario_dict = {"alarm": 3.1,
                     "audio": 3.2,
                     "iot": 3.3}
    for scenario, model_version in scenario_dict.items():
        pretrain_massive(num=-1,
                         model_version=model_version,
                         labels_version="00",
                         scenario=scenario,
                         ger_num=-1)

def pretrain_experiment6():
    """test pretrain model 2.0 fintuned on massive::scenarios under fewshot setting
    max_epoch=300 with early stopping"""
    model_versions = [6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7]

    for model_version in model_versions:
        pretrain_massive(num=-1,
                         model_version=model_version,
                         labels_version="00",
                         scenario="alarm",
                         ger_num=-1)


if __name__ == '__main__':
    pretrain_experiment6()
