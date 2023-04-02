import argparse
import math

import numpy as np
import pandas as pd
import torch
from promptsource.templates import DatasetTemplates
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from evaluation.sensitivity_evaluator import RougeRandom, AccuracyRandom
from evaluation.tasks.multilingual_datasets_index import get_datasets_by_lang
from evaluation.tasks.promptsource_task import PromptsourceDatasetTask

parser = argparse.ArgumentParser()

parser.add_argument("--model_names_or_paths", default="t5-small", type=str,
                    help="Coma-separated list of evaluated models' identifiers")
parser.add_argument("--use_cache", type=str, default="True", choices=('True', 'False'),
                    help="Whether to use cached predictions, if available.")
parser.add_argument("--firstn", type=int, default=500,
                    help="If given, a number of samples from dataset to evaluate on.")
parser.add_argument("--metric", default="ROUGE", type=str,
                    help="A metric to compute informative difference with. Must be one of the implemented metrics:"
                         "'ROUGE', 'Accuracy'.")
parser.add_argument("--max_input_length", default=None, type=int,
                    help="Maximum length of permitted model input. Longer inputs will be skipped to avoid OOM.")
parser.add_argument("--bootstrap", default="True", type=str,
                    help="Whether to collect a set of results over random subsets of predictions. Defaults to True.")
parser.add_argument("--langs", default="ru,pl", type=str,
                    help="Coma-separated list of languages to evaluate tasks on.")
parser.add_argument("--tasks", default="None", type=str,
                    help="Coma-separated list of tasks to evaluate tasks on.")

args = parser.parse_args()

args.use_cache = args.use_cache == "True"
args.bootstrap = args.bootstrap == "True"

results = {}

# eval iteration
for model_name_or_path in args.model_names_or_paths.split(","):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path,
                                                  # device_map=device_map,
                                                  # device_map="auto",  # TODO set for multi-GPU evaluation
                                                  # max_memory=max_memory_mapping
                                                  ).to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    results[model_name_or_path] = {}

    for lang in args.langs.split(","):

        lang_datasets = get_datasets_by_lang(lang)
        if args.tasks != "None":
            # subset evaluation datasets to the explicitly passed tasks
            lang_datasets = {task_id: dataset for task_id, dataset in lang_datasets.items()
                             if task_id in args.tasks.split(",")}
        for task_id, dataset in lang_datasets.items():
            for template_name in DatasetTemplates(task_id).all_template_names:
                task = PromptsourceDatasetTask(promptsource_id=task_id,
                                               prompts_template=template_name,
                                               hf_dataset=dataset)
                if args.metric == "ROUGE":
                    evaluator = RougeRandom(task,
                                            bootstrap=args.bootstrap,
                                            max_input_length=args.max_input_length,
                                            firstn=args.firstn if args.firstn else None)
                elif args.metric == "Accuracy":
                    evaluator = AccuracyRandom(task,
                                               bootstrap=args.bootstrap,
                                               max_input_length=args.max_input_length)
                else:
                    raise ValueError("Unknown metric: %s" % args.metric)

                # a list of results if args.bootstrap, a single prediction otherwise
                performance, _ = evaluator.evaluate_by_strategy(model,
                                                                tokenizer,
                                                                strategy="random",
                                                                use_cache=args.use_cache)
                if not args.bootstrap:
                    # unify the format, so we have a single result formatting
                    performance_to_print = "{:.5f}".format(performance)
                else:
                    mean = sum(performance) / len(performance)
                    q_lower = np.quantile(performance, q=0.025)
                    q_upper = np.quantile(performance, q=0.975)
                    broader_q = max((math.fabs(mean - q_lower), math.fabs(mean - q_upper)))

                    performance_to_print = "{:.5f}±{:.5f}".format(mean, broader_q)

                print("{}\t{}\t{}\t{}\t".format(model_name_or_path,
                                                    task.promptsource_id,
                                                    template_name,
                                                    performance_to_print))
                result_key = "%s-%s" % (task.promptsource_id, template_name)
                results[model_name_or_path][result_key] = performance_to_print

    pd.DataFrame(results).to_csv("%s_multilingual_evaluation.tsv" % model_name_or_path.split("/")[-1], sep="\t")
