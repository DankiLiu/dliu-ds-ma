from nltk.parse.corenlp import CoreNLPServer


def corenlp_server_start(path_to_jar=None,
                         path_to_models_jar=None):
    # Only in testing phase
    path_to_jar = '/home/danki/Studium/Thesis/stanford_parser/jars/stanford-corenlp-4.5.0.jar'
    path_to_models_jar = '/home/danki/Studium/Thesis/stanford_parser/jars/stanford-corenlp-4.5.0-models.jar'
    if not path_to_jar:
        path_to_jar = input("input the path to stanford-corenlp-4.5.0.jar\n")
    if not path_to_models_jar:
        path_to_models_jar = input("input the path to stanford-corenlp-4.5.0-model.jar\n")
    # Server
    server = CoreNLPServer(path_to_jar=path_to_jar,
                           path_to_models_jar=path_to_models_jar)
    print("Starting corenlp server ...")
    server.start()
    return server


def corenlp_server_stop(server):
    server.stop()
    print("Corenlp server stopped...")
