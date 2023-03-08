class Metric:
    def __init__(self, model_name, sample_num, metrics=None):
        self.model_name = model_name
        # in bymodel, sample_num is the number of examples
        # in bylabel, sample_num is the number of occurrences of every label
        self.sample_num = sample_num
        if not metrics:
            self.cor, self.par, self.inc, self.mis, self.spu = 0, 0, 0, 0, 0
        else:
            self.cor, self.par, self.inc, self.mis, self.spu = \
                metrics[0], metrics[1], metrics[2], metrics[3], metrics[4]

    def set_metrics(self, cor, par, inc, mis, spu):
        self.cor = cor
        self.par = par
        self.inc = inc
        self.mis = mis
        self.spu = spu

    def normalization(self):
        if self.sample_num == 0:
            return
        self.cor = self.cor/self.sample_num
        self.par = self.par/self.sample_num
        self.inc = self.inc/self.sample_num
        self.mis = self.mis/self.sample_num
        self.spu = self.spu/self.sample_num


class MetricByLabel(Metric):
    def __init__(self, model_name, label_name, num, metrics=None):
        super().__init__(model_name, num, metrics)
        self.label_name = label_name

    def to_pandas_df(self):
        # construct a pandas dataframe from metrics and return it
            #
        pass

    @classmethod
    def create_data_from_file(cls, model_name, label_name, path):
        # if file does not exist, throw exception
        import os
        if not os.path.exists(path):
            FileExistsError(f"{path} dose not exist")

        # read model name, num, metrics from file and create a MetricByModel instance
        import pandas as pd
        df = pd.read_csv(path)  # read whole file
        df_model = df.loc[(df['model_name'] == model_name) & (df['label_name'] == label_name)]
        num = df_model.iloc[0]["num"]
        cor = df_model.iloc[0]["correct"]
        par = df_model.iloc[0]["partial"]
        inc = df_model.iloc[0]["incorrect"]
        mis = df_model.iloc[0]["missing"]
        spu = df_model.iloc[0]["spurious"]
        # print(f"this file contains metrics with {num} examples for each label")
        # print("here print the number of rows, should be one")
        return cls(model_name, label_name, num, [cor, par, inc, mis, spu])


class MetricByModel(Metric):
    def __init__(self, model_name, sample_num, metrics=None):
        super().__init__(model_name, sample_num, metrics)

    def store_to_file(self):
        pass

    def to_pandas_df(self):
        # construct a pandas dataframe from metrics and return it
            #
        pass

    @classmethod
    def create_data_from_file(cls, model_name, path):
        # given metrics file path, load metrics from file and return a class
        # if file does not exist, throw exception
        import os
        if not os.path.exists(path):
            FileExistsError(f"{path} dose not exist")
        # read model name, num, metrics from file and create a MetricByModel instance
        import pandas as pd
        df = pd.read_csv(path)  # read whole file
        df_model = df.loc[df['model_name'] == model_name].mean(axis=0)   # df rows of given model name
        print(f"df_model after mean {df_model}")
        num = df_model["num"]
        print(f"this file contains metrics with {num} examples")
        cor = df_model["correct"]
        par = df_model["partial"]
        inc = df_model["incorrect"]
        mis = df_model["missing"]
        spu = df_model["spurious"]
        return cls(model_name, num, [cor, par, inc, mis, spu])
