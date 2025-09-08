from metrics.dp_metrics import DPMetric

class DPGAP(DPMetric):

    def __init__(self, sensitive_dataset, public_model, epsilon):

        super().__init__(sensitive_dataset, public_model, epsilon)

    def cal_metric(self):

        pass