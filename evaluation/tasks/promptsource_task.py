import abc
from typing import Union, Optional, Sequence

from datasets import Dataset
from promptsource.templates import DatasetTemplates

from evaluation import config
from evaluation.tasks.task import Task


class PromptsourceDatasetTask(Task, abc.ABC):
    promptsource_id: str
    dataset: Union[Dataset, None] = None

    def __init__(self,
                 promptsource_id: str,
                 prompts_template: str,
                 hf_dataset: Dataset):
        super().__init__()
        self.promptsource_id = promptsource_id
        dataset_templates = DatasetTemplates(self.promptsource_id)
        template = dataset_templates[prompts_template]
        self.label = promptsource_id.replace("/", "_") + "-" + prompts_template.replace(" ", "_")

        if config.firstn is not None:
            hf_dataset = hf_dataset.select(range(config.firstn))

        self.data = [(*template.apply(sample), None) for sample in hf_dataset]  # type: ignore
