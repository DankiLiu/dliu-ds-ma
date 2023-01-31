class MUC5:
    def __init__(self, cor, par, inc, mis, spu):
        self.cor = cor
        self.par = par
        self.loose_cor = self.cor + self.par
        self.inc = inc
        self.mis = mis
        self.spu = spu
        self.actual = cor + par + inc + spu
        self.possible = cor + par + inc + mis
        self._acc = self.acc()
        self._f = self.f()

    # todo: a loose evaluation of f1
    def f(self, b=1.0):
        recall = self.recall
        precision = self.precision
        numerator = (b * b + 1.0) * precision * recall
        denominator = (b * b * precision) + recall
        if denominator == 0:
            return 0
        return numerator / denominator

    def acc(self, is_loose=True):
        total = sum([self.cor, self.par, self.inc, self.mis, self.spu])
        if is_loose:
            return 0 if total == 0 else (self.cor + self.par) / total
        return 0 if total == 0 else self.cor / total

    @property
    def recall(self):
        numerator = self.cor + (self.par * 0.5)
        denominator = float(self.possible)
        if numerator == 0:
            return 0
        return numerator / denominator

    @property
    def precision(self):
        numerator = self.cor + (self.par * 0.5)
        denominator = float(self.actual)
        if denominator == 0:
            return 0
        return numerator / denominator


