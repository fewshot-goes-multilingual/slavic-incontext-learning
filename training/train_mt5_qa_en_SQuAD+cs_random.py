import json
from typing import List

from adaptor.adapter import Adapter
from adaptor.evaluators.generative import BLEU
from adaptor.lang_module import LangModule
from adaptor.schedules import ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from datasets import load_dataset

from priming_objective import Priming

training_arguments = AdaptationArguments(output_dir="train_dir_SQuAD_random_large",
                                         learning_rate=2e-5,  # we set LR=2e-4 for pre-training experiments
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         # stopping_strategy=StoppingStrategy.NUM_STEPS_TOTAL,
                                         do_train=True,
                                         do_eval=True,
                                         warmup_steps=1000,
                                         max_steps=10000,
                                         gradient_accumulation_steps=30,
                                         eval_steps=500,
                                         logging_steps=10,
                                         save_steps=500,
                                         num_train_epochs=5,
                                         evaluation_strategy="steps",
                                         save_total_limit=10,
                                         stopping_patience=10)
eval_examples = 200

# priming
num_demonstrations = 3


def _construct_priming_prompt(previous_examples: List[str], current_example: str) -> str:
    return " ".join(previous_examples + [current_example])


lang_module = LangModule("google/mt5-large")

# priming
per_type_examples = {}

qa_en = load_dataset("squad")
qa_train = qa_en["train"].filter(lambda entry: len(entry["context"]) < 2000)

val_metrics = [BLEU(**{"additional_sep_char": "▁"})]

# SQuAD QA dataset & objective:


def _get_en_qa_categories(data) -> List[str]:
    return [question.split()[0] if not question.startswith("To")
            else " ".join(question.split()[:2])
            for question in data["question"]]


q_answering_en = Priming(lang_module,
                         max_eval_samples=eval_examples,
                         demos_selection_strategy="random",
                         texts_or_path=qa_train["question"],
                         text_pair_or_path=qa_train["context"],
                         val_texts_or_path=qa_en["validation"]["question"][-eval_examples:],
                         val_text_pair_or_path=qa_en["validation"]["context"][-eval_examples:],
                         labels_or_path=[a["text"][0] for a in qa_train["answers"]],
                         val_labels_or_path=[a["text"][0] for a in qa_en["validation"]["answers"]][-eval_examples:],
                         train_question_categories=_get_en_qa_categories(qa_train),
                         val_question_categories=_get_en_qa_categories(qa_en["validation"])[-eval_examples:],
                         batch_size=1,
                         val_evaluators=val_metrics,
                         # val_evaluators=val_metrics,
                         source_lang_id="en",
                         objective_id="AQA-en")

# Czech data & objective

squad_cs_dataset = json.load(open("czech_squad_4-sents.json"))

skipped = 0

questions_cs = []
contexts_cs = []
answers_cs = []
categories_cs = []

for i, entry in squad_cs_dataset.items():
    if len(entry["context"]) > 800:
        skipped += 1
        continue

    questions_cs.append(entry["question"])
    contexts_cs.append(entry["context"])
    answers_cs.append(entry["answers"]["text"][0])
    categories_cs.append(entry["answer_type"])

print("Skipped cs examples: %s" % skipped)

q_answering_cs = Priming(lang_module,
                         max_eval_samples=eval_examples,
                         demos_selection_strategy="random",
                         texts_or_path=questions_cs[:-eval_examples],
                         text_pair_or_path=contexts_cs[:-eval_examples],
                         val_texts_or_path=questions_cs[-eval_examples:],
                         val_text_pair_or_path=contexts_cs[-eval_examples:],
                         labels_or_path=answers_cs[:-eval_examples],
                         val_labels_or_path=answers_cs[-eval_examples:],
                         train_question_categories=categories_cs[:-eval_examples],
                         val_question_categories=categories_cs[-eval_examples:],
                         batch_size=1,
                         val_evaluators=val_metrics,
                         source_lang_id="cs",
                         objective_id="SQUAD-cs")

schedule = ParallelSchedule(objectives=[q_answering_en, q_answering_cs],
                            args=training_arguments)

adapter = Adapter(lang_module, schedule, args=training_arguments)
adapter.train()
