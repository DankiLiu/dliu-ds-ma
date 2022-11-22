from parse.parsing_evaluation import testing

from gpt3.gpt3jointslu import gpt3jointslu
from gpt3.gpt3_util import read_output_from_file
from evaluation.evaluation import muc_5_metrics

if __name__ == '__main__':
    # gpt3jointslu(1)
    muc_5_metrics(num=5, model="parsing")