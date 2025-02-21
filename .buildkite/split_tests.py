import argparse
import os
from random import shuffle


def foo(input_file, n_splits=None):

    if n_splits is None:
        n_splits = 2

    with open(input_file) as fp:
        data = fp.read().splitlines()
        data = shuffle(data)

    n_items = len(data)
    n_items_per_split = n_items // n_splits
    reminder = n_items % n_splits
    n_items = [n_items_per_split + int(idx < reminder) for idx in range(n_splits)]

    splitted = {}
    current = 0
    for idx in range(n_splits):
        end = current + n_items[idx]
        splitted[idx] = data[current: end]
        current = end

    return splitted


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Where to store the list of tests to run"
    )
    args = parser.parse_args()

    fns = os.listdir(args.input_dir)
    for fn in fns:
        if fn.endswith("_test_list.txt"):

            splitted = foo(os.path.join(args.input_dir, fn))

            import json
            with open(os.path.join(args.input_dir, f"splitted_{fn.replace('.txt', '.json')}"), "w") as fp:
                json.dump(splitted, fp, indent=4)
