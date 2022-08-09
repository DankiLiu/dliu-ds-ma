import parse.util as util
from data.data_processing import jointslu_per_line
from parse.nltk_parser import dependency_parsing


def corenlp_parse(text):
    # start server
    # server = util.corenlp_server_start()
    # parse each sentence

    dependency_parsing(text=text)
    # stop server
    # util.corenlp_server_stop(server)


def main():
    with open("../data/sample.iob") as f:
        lines = f.readlines()

    for line in lines:
        sentence, words, labels = jointslu_per_line(line)
        print(f"sentence {sentence}")
        print(f"words    {words}")
        print(f"labels   {labels}")
        corenlp_parse(sentence)
        break

if __name__ == '__main__':
    main()