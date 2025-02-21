import argparse
import yaml
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--n_splits", type=int, required=False, default=1, help="Where to store the list of tests to run"
    # )
    # parser.add_argument(
    #     "--input_file", type=str, required=True, help="Where to store the list of tests to run"
    # )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Where to store the list of tests to run"
    )
    args = parser.parse_args()

    with open(".buildkite/config2.yml") as fp:
        data = fp.read()
    with open(args.output_file, "w") as fp:
        fp.write(data)

    config = dict()
    config["steps"] = []
    job = dict()
    job["label"] = "dummy"
    job["commands"] = ["pwd", "ls -la"]
    config["steps"].append(job)

    folder = ".buildkite"

    with open(os.path.join(folder, "generated_config.yml"), "w") as f:
        f.write(yaml.dump(config, sort_keys=False, default_flow_style=False))

    # with open(args.input_file) as fp:
    #     data = fp.read().splitlines()
    #
    # n_splits = args.n_splits
    #
    # n_items = len(data)
    # n_items_per_split = n_items // n_splits
    # reminder = n_items % n_splits
    # n_items = [n_items_per_split + int(idx < reminder) for idx in range(n_splits)]
    # # print(n_items)
    #
    # final = {}
    # current = 0
    # for idx in range(n_splits):
    #     end = current + n_items[idx]
    #     final[idx] = data[current: end]
    #     current = end
    #
    # import json
    # with open(args.output_file, "w") as fp:
    #     json.dump(final, fp, indent=4)
