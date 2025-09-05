import os
import sys
import argparse
import random

from metrics.dpfid import DPFID
from metrics.dpgap import DPGAP

def main(args):


    return


if __name__ == '__main__':

    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--public_model', default="")
    parser.add_argument('--sensitive_dataset', '-sd', default="cifar10_32")
    parser.add_argument('--epsilon', '-e', default="0.01")
    parser.add_argument('--metrics', '-m', default="DPGap")
    parser.add_argument('--exp_description', '-ed', default="")

    args = parser.parse_args()

    main(args)