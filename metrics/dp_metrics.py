

class DPMetric(object):


    def __init__(self, sensitive_dataset, public_model, epsilon):

        self.sensitive_dataset = sensitive_dataset
        self.public_model = public_model
        self.epsilon = epsilon

    def variant(self):

        return sensitive_dataset, variant_sensitive_dataset

    def cal_metric(self):

        pass
