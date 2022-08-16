import parse.util as util
from data.data_processing import jointslu_per_line
import parse.nltk_parser as np


def corenlp_parse(text):
    # pos parsing
    np.part_of_speech_parsing(text=text)
    # ner parsing
    np.name_entity_recognition(text=text)
    # dependency parsing
    np.dependency_parsing(text=text)


def main():
    # Start server if not started
    import os
    os.environ['JAVAHOME'] = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
    if not util.server_is_running("http://localhost:9000/"):
        util.corenlp_server_start()

    with open("../data/sample.iob") as f:
        lines = f.readlines()

    for line in lines:
        sentence, words, labels = jointslu_per_line(line)
        print(f"sentence {sentence}")
        print(f"words    {words}")
        print(f"labels   {labels}")
        corenlp_parse(sentence)
        if_continue = input("press enter to continue, 'n' to stop")
        if if_continue == "":
            continue
        else:
            break


if __name__ == '__main__':
    main()