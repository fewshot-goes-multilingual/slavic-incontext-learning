from typing import Dict

from datasets import Dataset


def get_datasets_by_lang(lang_id: str, training_split: bool = False) -> Dict[str, Dataset]:

    if lang_id == "cs":
        from evaluation.tasks.cs.customized_datasets import get_all_datasets
        return get_all_datasets(training_split)
    elif lang_id == "ru":
        from evaluation.tasks.ru.customized_datasets import get_all_datasets
        return get_all_datasets(training_split)
    elif lang_id == "pl":
        from evaluation.tasks.pl.customized_datasets import get_all_datasets
        return get_all_datasets(training_split)
    else:
        raise ValueError("No tasks for language %s." % lang_id)


def get_dataset_by_task_id(task_id: str, lang_id: str, training_split: bool = False) -> Dataset:
    return get_datasets_by_lang(lang_id, training_split)[task_id]
