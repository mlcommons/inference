import argparse
import json
import multiprocessing
import pickle
import queue
import re
import timeit

import pandas as pd
from tqdm import tqdm

from mxeval.execution import check_correctness as check_correctness_python
from mxeval.execution import (
    check_correctness_cpp,
    check_correctness_csharp,
    check_correctness_go,
    check_correctness_java,
    check_correctness_javascript,
    check_correctness_kotlin,
    check_correctness_perl,
    check_correctness_php,
    check_correctness_ruby,
    check_correctness_scala,
    check_correctness_swift,
    check_correctness_typescript,
)


def postprocess_golang(code: str) -> str:
    multi_line_imports = re.compile(
        r"^import \(\n(.+)((?:\n.+)+)\n\)", re.MULTILINE)
    line_imports = re.compile(r"^import \".*\"")
    func_main = re.compile(r"^func main.*^}", re.MULTILINE | re.DOTALL)

    code = code.replace("package main", "")  # Remove package main
    code = multi_line_imports.sub("", code)
    code = line_imports.sub("", code)
    code = func_main.sub("", code)

    return code


def postprocess_scala(code: str) -> str:
    code = code.replace("object Main extends App {", "")
    code = "".join(code.splitlines(True)[:-1])
    return code


def postprocess_python(code: str) -> str:
    return code.lstrip()


def worker(inp_queue, out_queue):
    while True:
        try:
            problem = inp_queue.get(timeout=5)
        except queue.Empty:
            break

        key = f"{problem['lang']}_{problem['entry_point']}"
        checker = eval(f"check_correctness_{problem['lang']}")

        problem["task_id"] = key
        problem["test"] = problem["test_code"]

        solution = problem["response"]

        try:
            solution = solution[:solution.index("```")]
        except ValueError:
            # Happens when a code block isn't closed properly
            pass

        if problem["lang"] == "go":
            solution = postprocess_golang(solution)
        elif problem["lang"] == "python":
            solution = postprocess_python(solution)
        elif problem["lang"] == "scala":
            solution = postprocess_scala(solution)

        # Mixtral likes escaping underscores for some reason, so let's remove
        # these
        solution = solution.replace("\\_", "_")

        # The evaluation script evaluates `code = prompt + solution + tests`
        # But Mixtral regenerates the prompt in its output, so we should remove
        # this
        problem["prompt"] = ""
        try:
            result = checker(problem, solution, timeout=20.0)
            out_queue.put(
                (key,
                 problem["lang"],
                    result["passed"],
                    result["result"],
                    problem["response"]))
        except Exception as e:
            print(e)
            out_queue.put(
                (key, problem["lang"], False, "", problem["response"]))


def evaluate_mbxp(results, n_workers):
    by_lang = {}
    for problem in results:
        by_lang.setdefault(problem["lang"], []).append(problem)

    inp_queue = multiprocessing.Queue()
    out_queue = multiprocessing.Queue()

    n_problems = 0

    for lang, problems in by_lang.items():
        if lang not in ["cpp", "python", "php",
                        "javascript", "ruby", "typescript"]:
            continue

        n_problems += len(problems)
        for problem in problems:
            inp_queue.put(problem)

    start = timeit.default_timer()
    workers = []
    for _ in range(n_workers):
        w = multiprocessing.Process(target=worker, args=(inp_queue, out_queue))
        w.start()
        workers.append(w)

    passes = {}
    n_passed = 0
    lang_passed = {}
    lang_counts = {}
    for i in tqdm(range(n_problems)):
        key, lang, passed, result, response = out_queue.get()
        passes[key] = {
            "passed": passed,
            "result": result,
            "response": response}
        n_passed += passed

        lang_passed.setdefault(lang, 0)
        lang_passed[lang] += passed

        lang_counts.setdefault(lang, 0)
        lang_counts[lang] += 1

    end = timeit.default_timer()
    print(f"Processed {n_problems} in {end - start}s")
    print(f"{100 * n_passed / n_problems : .02f}% pass@1")
    print(lang_passed, lang_counts)
    with open("evaluated_test.json", "w") as f:
        json.dump(passes, f, indent=2)

    return 100 * n_passed / n_problems
