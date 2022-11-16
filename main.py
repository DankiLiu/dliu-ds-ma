from parse.parsing_evaluation import testing

from gpt3.gpt3jointslu import gpt3jointslu
from gpt3.gpt3_util import read_output_from_file
from util import read_keys_from_json

if __name__ == '__main__':
    # gpt3jointslu(1)
    values = read_keys_from_json("data/jointslu/gpt3/gpt3_output.json",
                                 "prediction",
                                 "std_gt")
    print(values)