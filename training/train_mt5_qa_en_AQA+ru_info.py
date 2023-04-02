import json
from typing import List

from adaptor.adapter import Adapter
from adaptor.evaluators.generative import BLEU
from adaptor.lang_module import LangModule
from adaptor.schedules import ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from datasets import load_dataset

from training.priming_objective import Priming

training_arguments = AdaptationArguments(output_dir="train_dir_AQA_info_large_ru",
                                         learning_rate=2e-5,  # we set LR=2e-4 for pre-training experiments
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         # stopping_strategy=StoppingStrategy.NUM_STEPS_TOTAL,
                                         do_train=True,
                                         do_eval=True,
                                         warmup_steps=1000,
                                         max_steps=10000,
                                         gradient_accumulation_steps=30,  # TODO: set
                                         eval_steps=500,  # TODO: set
                                         logging_steps=10,
                                         save_steps=500,
                                         num_train_epochs=5,
                                         evaluation_strategy="steps",
                                         save_total_limit=10,
                                         stopping_patience=10)
eval_examples = 200  # TODO set

# priming
num_demonstrations = 3


def _construct_priming_prompt(previous_examples: List[str], current_example: str) -> str:
    return " ".join(previous_examples + [current_example])


# lang_module = LangModule("google/mt5-small")  # TODO set
# lang_module = LangModule("gaussalgo/mt5-base-priming-QA_en-cs")
# lang_module = LangModule("google/mt5-base")
lang_module = LangModule("google/mt5-large")

# priming
per_type_examples = {}

qa_en = load_dataset("adversarial_qa", "adversarialQA")
qa_train = qa_en["train"].filter(lambda entry: len(entry["context"]) < 2000)

val_metrics = [BLEU(**{"additional_sep_char": "▁"})]

# Adversarial QA dataset & objective:


def _get_firstword_categories(data) -> List[str]:
    return [question.split()[0] if not question.startswith("To")
            else " ".join(question.split()[:2])
            for question in data["question"]]


q_answering_en = Priming(lang_module,
                         max_eval_samples=eval_examples,
                         demos_selection_strategy="informative",  # TODO set
                         texts_or_path=qa_train["question"],
                         text_pair_or_path=qa_train["context"],
                         val_texts_or_path=qa_en["validation"]["question"][-eval_examples:],
                         val_text_pair_or_path=qa_en["validation"]["context"][-eval_examples:],
                         labels_or_path=[a["text"][0] for a in qa_train["answers"]],
                         val_labels_or_path=[a["text"][0] for a in qa_en["validation"]["answers"]][-eval_examples:],
                         train_question_categories=_get_firstword_categories(qa_train),
                         val_question_categories=_get_firstword_categories(qa_en["validation"])[-eval_examples:],
                         batch_size=1,
                         val_evaluators=val_metrics,
                         # val_evaluators=val_metrics,
                         source_lang_id="en",
                         objective_id="AQA-en")

qa_ru = load_dataset("sberquad")
qa_ru_train = qa_ru["train"].filter(lambda entry: len(entry["context"]) < 800)


skipped = 0

q_answering_ru = Priming(lang_module,
                         max_eval_samples=eval_examples,
                         demos_selection_strategy="informative",  # TODO set
                         texts_or_path=qa_ru_train["question"],
                         text_pair_or_path=qa_ru_train["context"],
                         val_texts_or_path=qa_ru["validation"]["question"][-eval_examples:],
                         val_text_pair_or_path=qa_ru["validation"]["context"][-eval_examples:],
                         labels_or_path=[a["text"][0] for a in qa_ru_train["answers"]],
                         val_labels_or_path=[a["text"][0] for a in qa_ru["validation"]["answers"]][-eval_examples:],
                         train_question_categories=_get_firstword_categories(qa_ru_train),
                         val_question_categories=_get_firstword_categories(qa_ru["validation"])[-eval_examples:],
                         batch_size=1,
                         val_evaluators=val_metrics,
                         # val_evaluators=val_metrics,
                         source_lang_id="ru",
                         objective_id="SQuAD-ru")

schedule = ParallelSchedule(objectives=[q_answering_en,
                                        q_answering_ru
                                        ],
                            args=training_arguments)

adapter = Adapter(lang_module, schedule, args=training_arguments)
adapter.train()
