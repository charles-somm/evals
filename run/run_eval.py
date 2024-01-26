import argparse
import logging
import os

from pathlib import Path

import evals
import openai
from evals.eval import Eval
from evals.registry import Registry

from dotenv import load_dotenv

load_dotenv()


def run_eval():
    # Define the arguments
    parser = argparse.ArgumentParser(description="Run evaluation script.")
    parser.add_argument(
        "--completion_fn",
        type=str,
        default="gpt-3.5-turbo-1106",
        help="Completion function name.",
    )
    parser.add_argument(
        "--eval", type=str, default="sommelia-prompt-04", help="Evaluation name."
    )
    parser.add_argument("--cache", type=bool, default=True, help="Use cache or not.")
    parser.add_argument("--seed", type=int, default=20220722)
    parser.add_argument("--record_path", type=str, default=None)
    parser.add_argument("--user", type=str, default="")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    # evaluation specifications
    registry = Registry()
    eval_spec = registry.get_eval(args.eval)

    eval_registry_path = Path(__file__).parent.parent / "evals" / "registry"

    # eval  object
    eval_class = registry.get_class(eval_spec)

    if args.max_samples is not None:
        evals.eval.set_max_samples(args.max_samples)

    openai.api_key = os.environ.get("OPENAI_API_KEY")
    logging.info(f"OpenAI API key: {openai.api_key}")
    completion_fn_instance = registry.make_completion_fn(args.completion_fn)
    eval: Eval = eval_class(
        completion_fns=[completion_fn_instance],
        # test_samples=eval_spec.args["samples_jsonl"],
        eval_registry_path=eval_registry_path,
        name=eval_spec.key,
        seed=args.seed,
    )

    # recorder
    eval_name = eval_spec.key
    run_spec = evals.base.RunSpec(
        completion_fns=[args.completion_fn],
        eval_name=eval_name,
        base_eval=eval_name.split(".")[0],
        split=eval_name.split(".")[1],
        run_config={
            "completion_fns": [args.completion_fn],
            "eval_spec": eval_spec,
            "max_samples": args.max_samples,
            "seed": args.seed,
        },
        created_by="charles",
    )
    # Record path dooesn't include file name. The file name is hard-coded.
    # Paths of the .jsonl and .csv files are the same.
    default_folder_path = (
        f"/mnt/c/Users/charl/OneDrive/sommelia/prompt_engineering/experiments/"
    )
    default_record_path = (
        default_folder_path
        # + "prout.jsonl"
        + f"{run_spec.run_id}_{args.completion_fn}_{args.eval}.jsonl"
    )
    record_path = default_record_path if args.record_path is None else args.record_path
    recorder = evals.CSVRecorder(record_path, run_spec)

    # run the evaluation
    from evals.data import get_jsonl

    result = eval.run(recorder=recorder)
    recorder.record_final_report(result)
    # lines = get_jsonl(record_path)
    # recorder.save_as_csv()


if __name__ == "__main__":
    run_eval()
