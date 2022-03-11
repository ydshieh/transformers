# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Checking utils for the docstrings."""

import argparse
import datetime
import json
import os
import random

from style_doc import _re_args, _re_list, _re_code, _re_doc_ignore, _re_returns, find_indent, is_empty_line, split_line_on_first_colon, _re_tip
import subprocess
import tempfile
import time
import warnings

from multiprocessing import Pool
from tqdm import tqdm


def extract_example_blocks(docstring):
    """Extract code example blocks in a docstring"""
    lines = docstring.split("\n")
    ### new_lines = []

    # Initialization
    current_paragraph = None
    current_indent = -1
    in_code = False
    param_indent = -1
    ### prefix = ""
    ### black_errors = []

    codes = []  ###

    # Special case for docstrings that begin with continuation of Args with no Args block.
    idx = 0
    while idx < len(lines) and is_empty_line(lines[idx]):
        idx += 1
    if (
        len(lines[idx]) > 1
        and lines[idx].rstrip().endswith(":")
        and find_indent(lines[idx + 1]) > find_indent(lines[idx])
    ):
        param_indent = find_indent(lines[idx])

    for idx, line in enumerate(lines):
        # Doing all re searches once for the one we need to repeat.
        list_search = _re_list.search(line)
        code_search = _re_code.search(line)

        # Are we starting a new paragraph?
        # New indentation or new line:
        new_paragraph = find_indent(line) != current_indent or is_empty_line(line)
        # List item
        new_paragraph = new_paragraph or list_search is not None
        # Code block beginning
        new_paragraph = new_paragraph or code_search is not None
        # Beginning/end of tip
        new_paragraph = new_paragraph or _re_tip.search(line)

        # In this case, we treat the current paragraph
        if not in_code and new_paragraph and current_paragraph is not None and len(current_paragraph) > 0:
            ### paragraph = " ".join(current_paragraph)
            ### new_lines.append(format_text(paragraph, max_len, prefix=prefix, min_indent=current_indent))
            current_paragraph = None

        if code_search is not None:
            if not in_code:
                current_paragraph = []
                current_indent = len(code_search.groups()[0])
                current_code = code_search.groups()[1]
                ### prefix = ""
                if current_indent < param_indent:
                    param_indent = -1
            else:
                current_indent = -1
                code = "\n".join(current_paragraph)
                if current_code in ["py", "python"]:
                    codes.append((idx - len(current_paragraph), idx, code))  ###
                    ### formatted_code, error = format_code_example(code, max_len, in_docstring=True)
                    ### new_lines.append(formatted_code)
                    ### if len(error) > 0:
                        ### black_errors.append(error)
                else:
                    pass  ###
                    ### new_lines.append(code)
                current_paragraph = None
            ### new_lines.append(line)
            in_code = not in_code

        elif in_code:
            current_paragraph.append(line)
        elif is_empty_line(line):
            current_paragraph = None
            current_indent = -1
            ### prefix = ""
            ### new_lines.append(line)
        elif list_search is not None:
            prefix = list_search.groups()[0]
            current_indent = len(prefix)
            current_paragraph = [line[current_indent:]]
        elif _re_args.search(line):
            ### new_lines.append(line)
            param_indent = find_indent(lines[idx + 1])
        ### elif _re_tip.search(line):
        ###    # Add a new line before if not present
        ###    if not is_empty_line(new_lines[-1]):
        ###        new_lines.append("")
        ###    new_lines.append(line)
        ###    # Add a new line after if not present
        ###    if idx < len(lines) - 1 and not is_empty_line(lines[idx + 1]):
        ###        new_lines.append("")
        elif current_paragraph is None or find_indent(line) != current_indent:
            indent = find_indent(line)
            # Special behavior for parameters intros.
            if indent == param_indent:
                # Special rules for some docstring where the Returns blocks has the same indent as the parameters.
                if _re_returns.search(line) is not None:
                    param_indent = -1
                    ### new_lines.append(line)
                ### elif len(line) < max_len:
                ###    new_lines.append(line)
                else:
                    intro, description = split_line_on_first_colon(line)
                    ### new_lines.append(intro + ":")
                    if len(description) != 0:
                        if find_indent(lines[idx + 1]) > indent:
                            current_indent = find_indent(lines[idx + 1])
                        else:
                            current_indent = indent + 4
                        current_paragraph = [description.strip()]
                        ### prefix = ""
            else:
                # Check if we have exited the parameter block
                if indent < param_indent:
                    param_indent = -1

                current_paragraph = [line.strip()]
                current_indent = find_indent(line)
                ### prefix = ""
        elif current_paragraph is not None:
            current_paragraph.append(line.lstrip())

    if current_paragraph is not None and len(current_paragraph) > 0:
        paragraph = " ".join(current_paragraph)
        ### new_lines.append(format_text(paragraph, max_len, prefix=prefix, min_indent=current_indent))

    ### return "\n".join(new_lines), "\n\n".join(black_errors)
    for (start, end, code) in codes:  ###
        print(code)  ###
        print("=" * 80)  ###
    return codes


def preprocess_code_example_block(code):
    """Pre-process a code example block"""
    code_lines = code.split("\n")

    # remove output lines
    code_lines = [x for x in code_lines if len(x.strip()) == 0 or x.lstrip().startswith(">>>") or x.lstrip().startswith("...")]

    # remove the 1st line if it is empty
    if code_lines and not code_lines[0].strip():
        code_lines = code_lines[1:]

    # check ">>>" and "..." formats
    for x in code_lines:
        if ">>>" in x:
            assert x.strip().startswith(">>> ")
        elif "..." in x:
            assert x.strip().startswith("... ") or x.lstrip() == "..."  # some possibly empty lines (in LED)

    # remove ">>>" and "..."
    code_lines = [x.strip().replace(">>> ", "").replace("... ", "") for x in code_lines]

    # deal with lines being "..."
    code_lines = [x if x != "..." else "" for x in code_lines]

    # put together into a code block
    code = "\n".join(code_lines)

    return code


def check_code_example_block(code, tmp_dir):
    """Check a code example block"""
    code = preprocess_code_example_block(code)
    result = {"status": "succeeded", "outputs": "", "elapsed": 0.0}

    # run the code example and capture the error if any
    if len(code.strip()) > 0:

        with open(os.path.join(tmp_dir, f"{hash(code)}.py"), "w", encoding="UTF-8") as fp:
            fp.write(code)
        s = datetime.datetime.now()
        result = subprocess.run(f'python {os.path.join(tmp_dir, f"{hash(code)}.py")}', shell=True, capture_output=True)
        e = datetime.datetime.now()
        elapsed = (e - s).total_seconds()
        status = "succeeded"
        outputs = result.stdout.decode("utf-8").replace("\r\n", "\n")
        if result.returncode != 0 and result.stderr:
            status = "failed"
            outputs = result.stderr.decode("utf-8").replace("\r\n", "\n")
        result = {
            "status": status,
            "outputs": outputs,
            "elapsed": elapsed,
        }

    result["time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return result


def check_docstring(docstring, tmp_dir):
    """Check code examples in `docstring` work"""
    # Make sure code examples in a docstring work
    results = {}

    code_example_blocks = extract_example_blocks(docstring)

    for (start, end, code_block) in code_example_blocks:
        result = check_code_example_block(code_block, tmp_dir)
        result["start"] = start
        result["end"] = end
        results[code_block] = result

    return results


def check_file_docstrings(code_file):
    """Check code examples in all docstrings in `code_file` work"""
    with open(code_file, "r", encoding="utf-8", newline="\n") as f:
        code = f.read()

    results = {}

    with tempfile.TemporaryDirectory() as tmp_dir:

        os.environ['TRANSFORMERS_CACHE'] = tmp_dir

        splits = code.split('\"\"\"')
        for i, s in enumerate(splits):
            if i % 2 == 0 or _re_doc_ignore.search(splits[i - 1]) is not None:
                continue
            result = check_docstring(s, tmp_dir)
            results.update(result)

    return results


def extract_code_example_blocks_from_docstring(docstring):

    codes = extract_example_blocks(docstring)
    return codes


def extract_code_example_blocks_from_file(code_file):

    with open(code_file, "r", encoding="utf-8", newline="\n") as f:
        code = f.read()

    codes = []
    splits = code.split('\"\"\"')
    for i, s in enumerate(splits):
        if i % 2 == 0 or _re_doc_ignore.search(splits[i - 1]) is not None:
            continue
        _codes = extract_code_example_blocks_from_docstring(s)
        codes.extend(_codes)

    return codes


def extract_code_example_blocks(*files):

    results = {}
    for file in files:
        # Treat folders
        if os.path.isdir(file):
            files = [os.path.join(file, f) for f in os.listdir(file)]
            files = [f for f in files if os.path.isdir(f) or f.endswith(".py")]
            results.update(extract_code_example_blocks(*files))
        elif file.endswith(".py"):
            codes = extract_code_example_blocks_from_file(file)
            if file not in results:
                results[file] = {}
            for (start, end, code) in codes:
                results[file][code] = {"start": start, "end": end}
        else:
            warnings.warn(f"Ignoring {file} because it's not a py file or a folder.")

    with open("code_example_blocks.json", "w", encoding="UTF-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=4)

    return results


def check_example_codes_multi_processing(codes):
    """Checks docstrings in all `files` and raises an error if there is any example which can't run.
    """

    data = []
    for file, file_info in codes.items():
        for block, block_info in file_info.items():
            data.append((file, block, block_info["start"], block_info["end"]))

    print(f"{len(data)} total code examples to check ...")

    # keep uniform
    for i in range(100):
        random.shuffle(data)

    s = time.time()

    batch_size = 32
    num_batches = len(data) // batch_size + int(len(data) % batch_size > 0)

    _results = []
    for batch_idx in tqdm(range(num_batches)):
        _s = time.time()
        batch = data[batch_size * batch_idx: batch_size * (batch_idx + 1)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.environ['TRANSFORMERS_CACHE'] = tmp_dir
            with Pool(8) as pool:
                batch_results = pool.starmap(check_code_example_block, [(block, tmp_dir) for _, block, _, _ in batch])
                _results.extend(batch_results)
        _e = time.time()
        print(f"Batch Timing: {_e - _s}")

    e = time.time()
    print(f'Total Timing: {e - s}')

    results = {}
    for (file, block, start, end), result in zip(data, _results):
        result["start"] = start
        result["end"] = end
        if file not in results:
            results[file] = {}
        if block not in results[file]:
            results[file][block] = {}
        results[file][block] = result

    with open("results-multi-processing.json", "w", encoding="UTF-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=4)

    return results


def check_doc_files(*files):
    """Checks docstrings in all `files` and raises an error if there is any example which can't run.
    """
    results = {}
    for file in files:
        # Treat folders
        if os.path.isdir(file):
            files = [os.path.join(file, f) for f in os.listdir(file)]
            files = [f for f in files if os.path.isdir(f) or f.endswith(".py")]
            results.update(check_doc_files(*files))
        elif file.endswith(".py"):
            _results = check_file_docstrings(file)
            if file not in results:
                results[file] = {}
            results[file].update(_results)
        else:
            warnings.warn(f"Ignoring {file} because it's not a py file or a folder.")

        if file in results:
            with open(file.replace("/", "-").replace("\\", "-").replace(".py", ".json"), "w", encoding="UTF-8") as fp:
                json.dump(results[file], fp, ensure_ascii=False, indent=4)

    with open("results.json", "w", encoding="UTF-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=4)

    return results


def main(*files, extract_only=False, multi_processing=False):

    if extract_only:
        return extract_code_example_blocks(*files)

    if multi_processing:
        codes = extract_code_example_blocks(*files)
        _results = check_example_codes_multi_processing(codes)
    else:
        _results = check_doc_files(*files)

    results = {}

    for file, res in _results.items():
        res = {code: result for code, result in res.items() if result["status"] != "succeeded"}
        if len(res) > 0:
            results[file] = res

    convert_json(_results, output="report.txt")
    convert_json(results, output="error_report.txt")

    if len(results) > 0:
        n_examples = sum(len(v) for v in results.values())
        raise ValueError(f"{n_examples} docstring examples in {len(results)} .py files should be fixed!")


def convert_json(json_report, output):

    with open(output, "w", encoding="UTF-8") as fp:
        for file_path in json_report:
            fp.write(file_path + "\n")
            fn = os.path.split(file_path)[-1]
            fp.write("=" * len(file_path) + "\n")
            for docstring, info in json_report[file_path].items():
                fp.write(f"file: {fn} | start: {info['start']} - end: {info['end']} | Elapsed Time: {info['elapsed']} | Time: {info['time']}")
                fp.write("\n\n")
                fp.write(docstring)
                fp.write("\n\n")
                indent = find_indent(docstring)
                fp.write(" " * indent + "Errors:\n\n")
                outputs = "\n".join([" " * (4 + indent) + x for x in info["outputs"].split("\n")])
                fp.write(outputs + "\n")
                fp.write("-" * len(file_path) + "\n")
            fp.write("\n")


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("files", nargs="+", help="The file(s) or folder(s) to check.")
    # parser.add_argument("--extract_only", action='store_true')
    # parser.add_argument("--multi_processing", action='store_true')
    # args = parser.parse_args()
    #
    # main(*args.files, extract_only=args.extract_only, multi_processing=args.multi_processing)



    # with open("results-multi-processing.json", "r", encoding="UTF-8") as fp:
    #     _results = json.load(fp)
    #
    # results = {}
    # for file, res in _results.items():
    #     res = {code: result for code, result in res.items() if result["status"] != "succeeded"}
    #     if len(res) > 0:
    #         results[file] = res
    #
    # results_pt = {
    #     file: res for file, res in results.items() if os.path.split(file)[-1].startswith("modeling_") and
    #         not (os.path.split(file)[-1].startswith("modeling_tf_") or os.path.split(file)[-1].startswith("modeling_flax_"))
    # }
    # results_tf = {file: res for file, res in results.items() if os.path.split(file)[-1].startswith("modeling_tf_")}
    # results_flax = {file: res for file, res in results.items() if os.path.split(file)[-1].startswith("modeling_tf_")}
    #
    # with open("errors.json", "w", encoding="UTF-8") as fp:
    #     json.dump(results, fp, ensure_ascii=False, indent=4)
    #
    # convert_json(results, "report-errors.txt")
    # convert_json(results_pt, "report-errors-pt.txt")
    # convert_json(results_tf, "report-errors-tf.txt")
    # convert_json(results_flax, "report-errors-flax.txt")



    with open("../errors.json", "r", encoding="UTF-8") as fp:
        errors = json.load(fp)
        failed_files = sorted(list(errors.keys()))
        failed_pt_files = [x for x in failed_files if os.path.split(x)[-1].startswith("modeling_") and not (os.path.split(x)[-1].startswith("modeling_tf_") or os.path.split(x)[-1].startswith("modeling_flax_"))]
        for x in failed_files:
            print(x)

    print("=" * 80)
    import glob
    root_dir = "../src/transformers/models/"
    files = [x.replace("../", "") for x in sorted(list(glob.iglob(root_dir + '**/**', recursive=True)))]
    pt_files = [x for x in files if os.path.split(x)[-1].startswith("modeling_") and not (os.path.split(x)[-1].startswith("modeling_tf_") or os.path.split(x)[-1].startswith("modeling_flax_"))]
    pt_files = [x for x in pt_files if not ("__pycache__" in x or x.endswith(".pyc"))]
    for x in pt_files:
        print(x)

    print("=" * 80)
    ok_pt_files = sorted(list(set(pt_files).difference(set(failed_pt_files))))
    for x in ok_pt_files:
        print(x)
