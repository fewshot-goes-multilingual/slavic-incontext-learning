import logging
import random
from typing import Iterable, Union, Dict, List, Optional

import torch
from adaptor.objectives.seq2seq import Sequence2Sequence
from transformers import BatchEncoding

logger = logging.getLogger()

priming_formats = {
    "QA": {"cs": "Otázka: %s Kontext: %s Odpověď:",
           "en": "Question: %s Context: %s Answer:",
           "ru": "Вопрос: %s Контекст: %s Отвечать:"}}


class Priming(Sequence2Sequence):

    def __init__(self, *args,
                 train_question_categories: Iterable[str],
                 max_eval_samples: int,
                 val_question_categories: Optional[Iterable[str]] = None,
                 min_num_demonstrations: int = 2,
                 max_num_demonstrations: int = 5,
                 demos_infer_batch_size: int = 32,
                 demos_selection_strategy: str = "hard",
                 difficulty_sample: int = 64,
                 max_input_length: int = 8000,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.train_question_categories = list(train_question_categories)
        self.val_question_categories = list(val_question_categories) if val_question_categories is not None else None

        self.min_num_demonstrations = min_num_demonstrations
        self.max_num_demonstrations = max_num_demonstrations
        self.demos_infer_batch_size = demos_infer_batch_size
        self.demos_selection_strategy = demos_selection_strategy
        self.difficulty_sample = difficulty_sample
        self.max_input_length = max_input_length
        self.max_eval_samples = max_eval_samples

    def _construct_qa_prompt(self, question: str, context: str) -> str:
        return priming_formats["QA"][self.source_lang_id] % (question, context)

    def _construct_demonstration(self, prompt: str, answer: str) -> str:
        return "%s %s " % (prompt, answer)

    def _construct_primed_prompt(self, primed_demonstrations: List[str], prompt: str) -> str:
        return " ".join(primed_demonstrations) + " " + prompt

    def forced_generation_score(self, input_texts: List[str], forced_output: str) -> torch.FloatTensor:
        inputs = self.tokenizer(input_texts, return_tensors="pt", padding="longest", truncation=True)
        inputs = inputs.to(self.compatible_head_model.device)

        with self.tokenizer.as_target_tokenizer():
            output_ids = self.tokenizer(forced_output, return_tensors="pt", padding="longest",
                                        truncation=True).input_ids.to(self.compatible_head_model.device)
            forced_outputs = self.compatible_head_model.prepare_decoder_input_ids_from_labels(output_ids)
            forced_outputs = forced_outputs.to(self.compatible_head_model.device)

        outputs = self.compatible_head_model(**inputs,
                                             decoder_input_ids=forced_outputs.expand(inputs.input_ids.shape[0], -1))
        output_log_probs = outputs.logits.log_softmax(-1)
        forced_output_logits = torch.gather(output_log_probs, -1,
                                            output_ids.expand(inputs.input_ids.shape[0], -1).unsqueeze(-1))
        forced_output_log_score = forced_output_logits.sum((-1, -2))
        # we do not need to normalize, as all the targets are the same <=> same length
        return forced_output_log_score.double().exp()

    def _pick_most_difficult_demo(self,
                                  selected_demos: List[str],
                                  next_demo_cands: List[str],
                                  predict_prompt: str,
                                  predicted_answer: str) -> int:
        with torch.no_grad():
            difficulties = torch.empty(0, device=self.compatible_head_model.device, dtype=torch.float)

            for batch_offset in range(0, len(next_demo_cands), self.demos_infer_batch_size):
                next_demo_cands_batch = next_demo_cands[batch_offset: batch_offset + self.demos_infer_batch_size]

                primed_prompts = [self._construct_primed_prompt(selected_demos + [demo], predict_prompt)
                                  for demo in next_demo_cands_batch]
                cands_difficulty = self.forced_generation_score(primed_prompts, predicted_answer)

                difficulties = torch.hstack((difficulties, cands_difficulty))

        assert difficulties.argmin() < len(next_demo_cands)

        return difficulties.argmin()

    def _get_inputs_iterator(self, split: str) -> Iterable[Union[BatchEncoding, Dict[str, torch.Tensor]]]:
        """
        Creates a default iterator over encodings with aligned input and output texts.
        :param split: Data split. `train` or `eval`.
        :return: Iterator of model input encodings.
        """
        # we materialize all samples in memory, so that we can heuristically pick the combinations
        questions, contexts, answers = (list(it) for it in self._per_split_iterators(split))
        question_categories = self.train_question_categories if split == "train" else self.val_question_categories

        assert len(questions) == len(contexts) == len(answers) == len(question_categories), \
            "Given numbers of questions, contexts and answers do not match."

        prompts = [self._construct_qa_prompt(q, c) for q, c in zip(questions, contexts)]

        features_batch = []
        cat_index = {cat: [i for i, sample_cat in enumerate(question_categories) if cat == sample_cat]
                     for cat in set(question_categories)}

        retrieved_samples = 0

        for idx, sample_category in enumerate(question_categories):
            if not cat_index[sample_category]:
                logger.warning("No samples within the category %s", sample_category)
                continue

            pred_prompt, pred_answer = prompts[idx], answers[idx]

            picked_demonstrations = []

            # a number of demonstrations is in the specified range
            expected_num_demonstrations = random.randint(self.min_num_demonstrations, self.max_num_demonstrations)

            while len(picked_demonstrations) < expected_num_demonstrations:
                if sum(map(len, picked_demonstrations)) > self.max_input_length:
                    logger.warning("Skipping too long prompt.")
                    break
                if self.demos_selection_strategy == "hard":
                    # pick the most difficult examples out of a sample
                    # we do not need to worry for picking up the predicted sample among demonstrations in hard strategy
                    if len(cat_index[sample_category]) <= 1:
                        # we can not construct informative demonstrations for categories of a single item
                        break

                    samples_idx = random.choices(cat_index[sample_category], k=self.difficulty_sample)
                    cand_demonstrations = [self._construct_demonstration(prompts[i], answers[i]) for i in samples_idx]
                    selected_index = self._pick_most_difficult_demo(picked_demonstrations, cand_demonstrations,
                                                                    pred_prompt, pred_answer)
                    picked_demonstrations.append(cand_demonstrations[selected_index])
                elif self.demos_selection_strategy == "informative":
                    if len(cat_index[sample_category]) <= 1:
                        # we can not construct informative demonstrations for categories of a single item
                        break
                    selected_cat_index = random.randint(1, len(cat_index[sample_category])-1)
                    selected_index = cat_index[sample_category][selected_cat_index]
                    if selected_index == idx:
                        # we do not want to expose the predicted sample in demonstrations
                        selected_index = cat_index[sample_category][selected_cat_index-1]
                    picked_demonstration = self._construct_demonstration(prompts[selected_index],
                                                                         answers[selected_index])
                    picked_demonstrations.append(picked_demonstration)
                elif self.demos_selection_strategy == "random":
                    # evaluation: do not infer samples' difficulty, pick randomly
                    selected_index = random.randint(1, len(prompts)-1)
                    if selected_index == idx:
                        # we do not want to expose the predicted sample in demonstrations
                        selected_index -= 1
                    picked_demonstration = self._construct_demonstration(prompts[selected_index],
                                                                         answers[selected_index])
                    picked_demonstrations.append(picked_demonstration)
                else:
                    raise ValueError("Unknown demon selection strategy: '%s'" % self.demos_selection_strategy)
            if len(picked_demonstrations) != expected_num_demonstrations:
                # we omit examples with none or only one demonstration in the category
                continue

            # encode a yielded batch
            primed_prompt = self._construct_primed_prompt(picked_demonstrations, pred_prompt)

            primed_prompt_encoding = self.tokenizer(primed_prompt, truncation=True)
            label_encoding = self.tokenizer(pred_answer, truncation=True)

            features_batch.append({"input_ids": primed_prompt_encoding.input_ids,
                                   "attention_mask": primed_prompt_encoding.attention_mask,
                                   "labels": label_encoding.input_ids})
            if len(features_batch) == self.batch_size:
                yield self.collator(features_batch)
                features_batch = []

            retrieved_samples += 1
            if split == "eval" and retrieved_samples >= self.max_eval_samples:
                # custom evaluation break - we need all samples in set to match categories,
                # but do not want to iterate them all
                break

        if features_batch:
            # yield last nonempty residual batch
            yield self.collator(features_batch)
