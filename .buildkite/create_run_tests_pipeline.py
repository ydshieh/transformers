import argparse
import yaml
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


COMMON_ENV_VARIABLES = {
    "OMP_NUM_THREADS": 1,
    "TRANSFORMERS_IS_CI": "true",
    "PYTEST_TIMEOUT": 120,
    "RUN_PIPELINE_TESTS": "false",
}


@dataclass
class Job:
    name: str
    additional_env: Dict[str, Any] = None
    docker_image: List[Dict[str, str]] = None
    install_steps: List[str] = None
    marker: Optional[str] = None

    def __post_init__(self):
        # Deal with defaults for mutable attributes.
        if self.additional_env is None:
            self.additional_env = {}
        if self.install_steps is None:
            self.install_steps = ["uv venv && uv pip install ."]

    def to_dict(self):
        env = COMMON_ENV_VARIABLES.copy()
        env.update(self.additional_env)

        env = [f"{k}={v}" for k,v in env.items()]

        # specific to BUILDKITE
        env.extend(["BUILDKITE_PARALLEL_JOB", "BUILDKITE_BRANCH"])

        test_file = f"test_preparation/splitted_{self.job_name}_test_list.json"
        # `parallelism` should correspond to the test split file computed previously
        with open(test_file) as fp:
            parallelism = len(json.load(fp))

        job = dict()
        job["label"] = self.job_name
        job["plugins"] = [
            {
                "docker#v5.12.0": {
                    "image": self.docker_image,
                    "always-pull": "true",
                    "mount-buildkite-agent": "true",
                    "environment": env
                }
            }
        ]
        job["parallelism"] = parallelism

        marker_cmd = f"-m '{self.marker}'" if self.marker is not None else ""

        job["commands"] = [
            "mkdir test_preparation",
            "buildkite-agent artifact download \"test_preparation/*\" test_preparation/ --step fetch_tests",
            "ls -la test_preparation",
            " && ".join(self.install_steps),
            "python -c \"import nltk; nltk.download('punkt', quiet=True)\"" if "example" in self.name else "echo Skipping",
            "du -h -d 1 \"$(pip -V | cut -d ' ' -f 4 | sed 's/pip//g')\" | grep -vE \"dist-info|_distutils_hack|__pycache__\" | sort -h | tee installed.txt || true",
            "pip list --format=freeze | tee installed.txt || true",
            # TODO: why failed
            "dpkg-query --show --showformat='${Installed-Size}\t${Package}\n' | sort -rh | head -25 | sort -h | awk '{ package=$2; sub(\".*/\", "", package); printf(\"%.5f GB %s\n\", $1/1024/1024, package)}' || true",
            f"TEST_SPLITS=$(python -c 'import os; import json; fp = open(\"{test_file}\"); data = json.load(fp); fp.close(); test_splits = data[os.environ[\"BUILDKITE_PARALLEL_JOB\"]]; test_splits = \" \".join(test_splits); print(test_splits);')",
            "echo \"$$TEST_SPLITS\"",
            f"python -m pytest {marker_cmd} pytest -n 8 -v $$TEST_SPLITS",
        ]

        return job

    @property
    def job_name(self):
        return self.name if ("examples" in self.name or "pipeline" in self.name or "pr_documentation" in self.name) else f"tests_{self.name}"


# JOBS
torch_job = Job(
    "torch",
    docker_image="huggingface/transformers-torch-light",
    marker="not generate",
)

pipelines_torch_job = CircleCIJob(
    "pipelines_torch",
    additional_env={"RUN_PIPELINE_TESTS": "true"},
    docker_image="huggingface/transformers-torch-light",
    marker="is_pipeline_test",
)

generate_job = Job(
    "generate",
    docker_image="huggingface/transformers-torch-light",
    marker="generate",
)

REGULAR_TESTS = [torch_job, generate_job]
ALL_TESTS = REGULAR_TESTS


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file", type=str, required=True, help="Where to store the list of tests to run"
    )
    args = parser.parse_args()

    config = dict()
    config["steps"] = []

    jobs = [k for k in ALL_TESTS if os.path.isfile(os.path.join("test_preparation" , f"{k.job_name}_test_list.txt"))]
    for job in jobs:
        config["steps"].append(job.to_dict())

    print(args.output_file)
    with open(args.output_file, "w") as f:
        f.write(yaml.dump(config, sort_keys=False, default_flow_style=False, width=float("inf")))
