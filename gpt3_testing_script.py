from evaluation.main import evaluate_bymodel, evaluate_slot, evaluate_intent
from main import test_model


def gpt3_testing(num, dataset, model_version, labels_version, scenario, ger_num=-1, gener=True):
    """Test parsing with jointslu, three times and calcualte """
    # test model and generate an output file
    gen = False
    model_eva = False
    intent_eva = False
    slot_eva = False

    if gener:
        gen = True
    else:
        model_eva = True
        intent_eva = True
        slot_eva = True

    if gen:
        test_model(model_name="gpt3",
                   num=num,
                   model_version=model_version,
                   dataset=dataset,
                   labels_version=labels_version,
                   scenario=scenario,
                   few_shot_num=-1)

    # evaluate bymodel three times
    if model_eva:
        evaluate_bymodel(sample_num=num,
                         dataset=dataset,
                         labels_version=labels_version,
                         generate=True,
                         scenario=scenario,
                         num_experiments=1,
                         model="gpt3")
    if slot_eva:
        evaluate_slot(sample_num=num,
                      dataset=dataset,
                      labels_version=labels_version,
                      generate=True,
                      scenario=scenario,
                      num_experiments=1,
                      model="gpt3")
    if intent_eva:
        evaluate_intent(sample_num=num,
                        dataset=dataset,
                        labels_version=labels_version,
                        generate=True,
                        scenario=scenario,
                        num_experiments=1,
                        model="gpt3")


def gpt3_experiment1():
    """test on ATIS using model 1.0 (zeroshot)"""
    zeroshot_withoutkeys = True
    zeroshot_withkeys = False
    if zeroshot_withoutkeys:
        gpt3_testing(num=-1,
                     dataset="jointslu",
                     model_version=1.0,
                     labels_version="02",
                     scenario=None,
                     ger_num=-1)
    if zeroshot_withkeys:
        gpt3_testing(num=-1,
                     dataset="jointslu",
                     model_version=1.0,
                     labels_version="02",
                     scenario=None,
                     ger_num=-1)


def gpt3_experiment2():
    """test on ATIS using model 2.0 (oneshot)"""
    oneshot_withoutkeys = True
    oneshot_withkeys = False
    if oneshot_withoutkeys:
        gpt3_testing(num=-1,
                     dataset="jointslu",
                     model_version=2.0,
                     labels_version="02",
                     scenario=None,
                     ger_num=-1)
    if oneshot_withkeys:
        gpt3_testing(num=-1,
                     dataset="jointslu",
                     model_version=2.0,
                     labels_version="02",
                     scenario=None,
                     ger_num=-1)


def gpt3_experiment3(scenario, model_version, gener):
    """test on MASSIVE scenarios using different models 1.0, """
    gpt3_testing(num=124,
                 dataset="massive",
                 model_version=model_version,
                 labels_version="00",
                 scenario=scenario,
                 ger_num=-1,
                 gener=gener)


if __name__ == '__main__':
    gpt3_experiment3("iot", 6.1, gener=False)