import argparse
import yaml
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Job:
    name: str
    docker_image: List[Dict[str, str]] = None

    def to_dict(self):

        job = dict()
        job["label"] = self.job_name
        job["plugins"] = [
            {
                "docker#v5.12.0": {
                    "image": self.docker_image,
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

        return job

    @property
    def job_name(self):
        return self.name if ("examples" in self.name or "pipeline" in self.name or "pr_documentation" in self.name) else f"tests_{self.name}"


# JOBS
torch_job = Job(
    "torch",
    docker_image="huggingface/transformers-torch-light",
    # marker="not generate",
    # parallelism=6,
)

REGULAR_TESTS = [torch_job]
ALL_TESTS = REGULAR_TESTS


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

    jobs = [k for k in ALL_TESTS if os.path.isfile(os.path.join("test_preparation" , f"{k.job_name}_test_list.txt"))]
    for job in jobs:
        config["steps"].append(job.to_dict())

    folder = ".buildkite"

    print(args.output_file)
    # with open(os.path.join(folder, args.output_file), "w") as f:
    with open(args.output_file, "w") as f:
        f.write(yaml.dump(config, sort_keys=False, default_flow_style=False, width=float("inf")))
