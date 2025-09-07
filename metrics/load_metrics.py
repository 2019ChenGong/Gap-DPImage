from metrics.dpfid import DPFID
from metrics.dpgap import DPGAP

def load_metrics(metrics_name):

    if metrics_name == "DPFID":
        metrics_model = DPFID()
    elif metrics_name == "DPGAP":
        metrics_model = DPGAP()
    else:
        print(f"Error: '{metrics_name}' is not a valid measure method.")
        return

    return metrics_model
    



