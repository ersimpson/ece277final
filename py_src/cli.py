from argparse import ArgumentParser
import sys
from pathlib import Path

from py_src.model import run, set_seed


def get_args():
    parser = ArgumentParser()
    default_data_path = (Path.home() / "ece277final_data").absolute()
    parser.add_argument("--data-path", type=str, default=str(default_data_path))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--batches", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.1)
    return parser.parse_args(sys.argv[1:])


def entrypoint():
    args = get_args() 
    set_seed(args.seed)
    run(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        batches=args.batches,
    )

if __name__ == "__main__":
    entrypoint()
