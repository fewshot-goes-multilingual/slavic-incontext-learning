import abc
import logging
import random
from typing import Optional, Tuple, List, Union

from adaptor.evaluators.evaluator_base import EvaluatorBase
from adaptor.evaluators.generative import ROUGE
from transformers import PreTrainedTokenizer, PreTrainedModel

from evaluation.evaluator import Evaluator
from evaluation.tasks.task import Task


logger = logging.getLogger()


class InformativeEvaluatorBase:

    def __init__(self,
                 task: Task,
                 num_demonstrations: int = 3,
                 firstn: Optional[int] = None,
                 bootstrap: bool = False,
                 max_input_length: Optional[int] = None,
                 reuse_last_run: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.task = task
        self.num_demonstrations = num_demonstrations
        self.firstn = firstn
        self.bootstrap = bootstrap
        self.max_input_length = max_input_length
        self.reuse_last_run = reuse_last_run

    @abc.abstractmethod
    def _compute(self, expected: List[str], actual: List[str]) -> float:
        pass

    def _compute_bootstrapped(self,
                              expected_all: List[str],
                              actual_all: List[str],
                              per_round_samples: int = 100,
                              repeats: int = 200) -> List[float]:
        assert len(expected_all) == len(actual_all), "Prediction lists' length do not match"

        evals = []
        while len(evals) < repeats:
            subset_idx = [random.randrange(len(expected_all)) for _ in range(per_round_samples)]
            expected_subset = [expected_all[idx] for idx in subset_idx]
            actual_subset = [actual_all[idx] for idx in subset_idx]

            evals.append(self._compute(expected_subset, actual_subset))

        return evals

    def evaluate_by_strategy(self,
                             model: PreTrainedModel,
                             tokenizer: PreTrainedTokenizer,
                             strategy: str,
                             eval_set: Optional[List[Tuple[str, str, str]]] = None,
                             use_cache: bool = True) -> Tuple[Union[List[float], float],
                                                              List[Tuple[str, str, str]]]:

        expected, actual, eval_set_new = Evaluator.collect_predictions(model, tokenizer, self.task,
                                                                       self.num_demonstrations, self.firstn,
                                                                       demo_selection_strategy=strategy,
                                                                       max_input_length=self.max_input_length,
                                                                       eval_set=eval_set,
                                                                       use_cache=use_cache)
        if "ru_" in self.task.label:
            logger.info("Found 'ru' in task label. Transliterating from Cyrillic. "
                        "This requires to have `transliterate` package installed.")
            from transliterate import translit

            expected = [translit(e, "ru", reversed=True) for e in expected]
            actual = [translit(a, "ru", reversed=True) for a in actual]

        if self.bootstrap:
            result = self._compute_bootstrapped(expected, actual)
        else:
            result = self._compute(expected, actual)

        return result, eval_set_new

    def get_per_sampling_performance(self,
                                     model: PreTrainedModel,
                                     tokenizer: PreTrainedTokenizer,
                                     use_cache: bool = True) -> Tuple[Union[List[float], float],
                                                                      Union[List[float], float]]:
        # we subset the predicted samples to ones having informative few-shots
        # - there's always less samples in 'informative' group
        informative_performance, eval_set = self.evaluate_by_strategy(model, tokenizer,
                                                                      strategy="cluster-random",
                                                                      use_cache=use_cache)
        random_performance, _ = self.evaluate_by_strategy(model, tokenizer,
                                                          strategy="random",
                                                          eval_set=eval_set,
                                                          use_cache=use_cache)

        return random_performance, informative_performance

    def __str__(self):
        return "%s_%s" % (self.task, super().__str__())


class RougeRandom(InformativeEvaluatorBase, ROUGE):

    def _compute(self, expected: List[str], actual: List[str]) -> Union[float, List[float]]:
        return self.evaluate_str(expected, actual)

    def __call__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, _) -> float:
        expected, actual, eval_set = Evaluator.collect_predictions(model, tokenizer, self.task,
                                                                   self.num_demonstrations, self.firstn,
                                                                   demo_selection_strategy="random",
                                                                   max_input_length=self.max_input_length,
                                                                   use_cache=self.reuse_last_run)
        random_performance = self._compute(expected, actual)

        return random_performance


class RougeInformative(RougeRandom, ROUGE):

    def __call__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, _) -> float:
        expected, actual, eval_set = Evaluator.collect_predictions(model, tokenizer, self.task,
                                                                   self.num_demonstrations, self.firstn,
                                                                   demo_selection_strategy="cluster-random",
                                                                   max_input_length=self.max_input_length,
                                                                   use_cache=self.reuse_last_run)
        info_performance = self._compute(expected, actual)

        return info_performance


class AccuracyRandom(RougeRandom, EvaluatorBase):

    def _compute(self, expected: List[str], actual: List[str]) -> Union[float, List[float]]:
        num_correct = sum([exp == act for exp, act in zip(expected, actual)])
        return num_correct / len(expected)


class AccuracyInformative(AccuracyRandom, RougeInformative, EvaluatorBase):
    pass
