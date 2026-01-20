from metrics.dpfid import DPFID
from metrics.dpgap import DPGAP
from metrics.pe_select import PE_Select
from metrics.dpprecision import DPPrecision

def load_metrics(metrics_name, sensitive_dataset, public_model, epsilon):

    if metrics_name == "DPFID":
        metrics_model = DPFID(sensitive_dataset, public_model, epsilon)
    elif metrics_name == "DPGAP":
        metrics_model = DPGAP(sensitive_dataset, public_model, epsilon)
    elif metrics_name == "PE-Select":
        metrics_model = PE_Select(sensitive_dataset, public_model, epsilon)
    elif metrics_name == "DPPrecision":
        metrics_model = DPPrecision(sensitive_dataset, public_model, epsilon)
    else:
        print(f"Error: '{metrics_name}' is not a valid measure method.")
        return

    return metrics_model
    



