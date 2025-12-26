import os

os.environ['HF_HOME'] = '/bigtemp/fzv6en/diffuser_cache'

import sys
import argparse
import random

from metrics.load_metrics import load_metrics
from metrics.load_sensitive_data import load_sensitive_data
from metrics.load_public_model import load_public_model

from metrics.dp_metrics import DPMetric



def main(args):

    sensitive_dataset, _ = load_sensitive_data(args.sensitive_dataset)

    public_model = load_public_model(args.public_model)

    metrics = load_metrics(args.metrics, sensitive_dataset, public_model, args.epsilon)

    results = metrics.cal_metric(args)

    return


if __name__ == '__main__':

    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--public_model', '-pm', default="stable-diffusion-v1-5", help="[stable-diffusion-2-1-base, \
         stable-diffusion-v1-5, stable-diffusion-v1-4, sstable-diffusion-2-base, dpimagebench-ldm]")
    parser.add_argument('--sensitive_dataset', '-sd', default="cifar10", help="[mnist, cifar10, octmnist, celeba, camelyon]")
    parser.add_argument('--epsilon', '-e', type=float, default=0.05)
    parser.add_argument('--metrics', '-m', default="DPFID", help="[DPFID, DPGAP]")
    parser.add_argument('--save_dir', '-s_d', default="exp/test", help="the path used to store the variant images")
    parser.add_argument('--exp_description', '-ed', default="")

    args = parser.parse_args()

    main(args)