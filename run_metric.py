import os
import sys
import argparse
import random

from metrics.load_metrics import load_metrics
from metrics.load_sensitive_data import load_sensitive_data
from metrics.load_public_model import load_public_model


def main(args):

    metrics = load_metrics(args.metrics)

    sensitive_dataset = load_sensitive_data(args.sensitive_dataset)

    public_model = load_public_model(args.public_model)

    results = metrics.cal_metric(args.sensitive_dataset, args.public_model)

    return


if __name__ == '__main__':

    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--public_model', default="stable-diffusion-2-1-base", help="[stable-diffusion-2-1-base, stable-diffusion-v1-5]")
    parser.add_argument('--sensitive_dataset', '-sd', default="cifar10", help="[mnist, cifar10, covidx, celeba, camelyon]")
    parser.add_argument('--epsilon', '-e', default="0.01")
    parser.add_argument('--metrics', '-m', default="DPFID", help="[DPFID, DPGAP]")
    parser.add_argument('--exp_description', '-ed', default="")

    args = parser.parse_args()

    main(args)