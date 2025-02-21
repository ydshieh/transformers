import argparse
import os


def foo(input_file, n_splits=None):

    if n_splits is None:
        n_splits = 2

    with open(input_file) as fp:
        data = fp.read().splitlines()

    n_items = len(data)
    n_items_per_split = n_items // n_splits
    reminder = n_items % n_splits
    n_items = [n_items_per_split + int(idx < reminder) for idx in range(n_splits)]

    splitted = {}
    current = 0
    for idx in range(n_splits):
        end = current + n_items[idx]
        final[idx] = data[current: end]
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

#
# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--n_splits", type=int, required=False, default=1, help="Where to store the list of tests to run"
#     )
#     parser.add_argument(
#         "--input_file", type=str, required=True, help="Where to store the list of tests to run"
#     )
#     parser.add_argument(
#         "--output_file", type=str, required=True, help="Where to store the list of tests to run"
#     )
#     args = parser.parse_args()
#
#     with open(args.input_file) as fp:
#         data = fp.read().splitlines()
#
#     n_splits = args.n_splits
#
#     n_items = len(data)
#     n_items_per_split = n_items // n_splits
#     reminder = n_items % n_splits
#     n_items = [n_items_per_split + int(idx < reminder) for idx in range(n_splits)]
#     # print(n_items)
#
#     final = {}
#     current = 0
#     for idx in range(n_splits):
#         end = current + n_items[idx]
#         final[idx] = data[current: end]
#         current = end
#
#     import json
#     with open(args.output_file, "w") as fp:
#         json.dump(final, fp, indent=4)
