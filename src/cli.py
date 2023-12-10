from argparse import ArgumentParser
import sys

from src.model import run, set_seed


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--save-freq", type=int, default=10)
    return parser.parse_args(sys.argv[1:])


def entrypoint():
    args = get_args() 
    set_seed(args.seed)
    run(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_freq=args.save_freq,
    )