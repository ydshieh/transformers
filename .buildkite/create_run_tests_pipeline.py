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
    job["plugins"] = [
        {
            "docker#v5.12.0": {
                "image": "huggingface/transformers-torch-light",
                "always-pull": "true",
                "mount-buildkite-agent": "true",
                "environment": [
                    "OMP_NUM_THREADS=1",
                    "BUILDKITE_PARALLEL_JOB",
                    "BUILDKITE_BRANCH",
                ]
            }
        }
    ]
    job["parallelism"] = 2
    job["commands"] = [
        "mkdir test_preparation",
        "buildkite-agent artifact download \"test_preparation/*\" test_preparation/ --step fetch_tests",
        "ls -la test_preparation",
        "echo \"pip install packages\"",
        "python -m pip install -U -e .",
        "TEST_SPLITS_2=$(python -c 'import os; import json; fp = open(\"test_preparation/splitted_shuffled_tests_torch_test_list.json\"); data = json.load(fp); fp.close(); test_splits = data[os.environ[\"BUILDKITE_PARALLEL_JOB\"]]; test_splits = \" \".join(test_splits); print(test_splits);')",
        "echo \"$TEST_SPLITS_2\"",
        # # 'TEST_SPLITS_2=$(python -c ''import os; import json; fp = open("test_preparation/splitted_shuffled_tests_torch_test_list.json"); data = json.load(fp); fp.close(); test_splits = data[os.environ["BUILDKITE_PARALLEL_JOB"]]; test_splits = " ".join(test_splits); print(test_splits);'')',
        # "python -m pytest -n 8 -v $$TEST_SPLITS_2",
    ]
    config["steps"].append(job)


    folder = ".buildkite"

    with open(os.path.join(folder, "generated_config.yml"), "w") as f:
        f.write(yaml.dump(config, sort_keys=False, default_flow_style=False, width=float("inf")))

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
