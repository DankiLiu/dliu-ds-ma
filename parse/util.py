import requests
from nltk.parse.corenlp import CoreNLPServer
import json

def corenlp_server_start(path_to_jar=None,
                         path_to_models_jar=None):
    # Only in testing phase
    path_to_jar = 'C:/Users/liuda/Documents/Files/StudiumInDtl/Masterarbeit/corenlp/stanford-corenlp-4.5.0/stanford-corenlp-4.5.0.jar'
    path_to_models_jar = 'C:/Users/liuda/Documents/Files/StudiumInDtl/Masterarbeit/corenlp/stanford-corenlp-4.5.0/stanford-corenlp-4.5.0-models.jar'
    if not path_to_jar:
        path_to_jar = input("input the path to stanford-corenlp-4.5.0.jar/n")
    if not path_to_models_jar:
        path_to_models_jar = input("input the path to stanford-corenlp-4.5.0-model.jar/n")
    # Server
    server = CoreNLPServer(path_to_jar=path_to_jar,
                           path_to_models_jar=path_to_models_jar)
    print("Starting corenlp server ...")
    server.start()
    return server


def corenlp_server_stop(server):
    server.stop()
    print("Corenlp server stopped...")


def server_is_running(url):
    try:
        page = requests.get(url)
        status_code = page.status_code
    except Exception as e:
        print(e)
        return False
    print(f"status code from {url} is {status_code}")
    if status_code == 200:
        print("server running, ready to parse")
        return True
    else:
        print("server not running")
        return False


def read_jointslu_labels():

    return None


def set_jointslu_labels(new_labels):
    data = {"jointslu_labels": new_labels}
    json.dump(data, "jointslu.json")
