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
        "echo \"$$TEST_SPLITS_2\"",
        "python -m pytest -n 8 -v $$TEST_SPLITS_2",
        # # 'TEST_SPLITS_2=$(python -c ''import os; import json; fp = open("test_preparation/splitted_shuffled_tests_torch_test_list.json"); data = json.load(fp); fp.close(); test_splits = data[os.environ["BUILDKITE_PARALLEL_JOB"]]; test_splits = " ".join(test_splits); print(test_splits);'')',
        # "python -m pytest -n 8 -v $$TEST_SPLITS_2",
    ]
    config["steps"].append(job)

    folder = ".buildkite"

    print(args.output_file)
    # with open(os.path.join(folder, args.output_file), "w") as f:
    with open(args.output_file, "w") as f:
        f.write(yaml.dump(config, sort_keys=False, default_flow_style=False, width=float("inf")))
