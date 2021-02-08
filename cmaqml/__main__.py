from .driver import run
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--validate', default=0, type=int)
parser.add_argument('--config', default='config.json', type=str)
args = parser.parse_args()

outf = run(args.config, args.validate)
