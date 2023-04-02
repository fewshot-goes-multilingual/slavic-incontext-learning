# Training New In-context Few-shot learners using QA

The `train*.py` scripts in this folder will run the training of in-context few-shot 
learners using QA datasets of the target languages.

Thanks to using [Adaptor objectives](https://github.com/gaussalgo/adaptor), 
all training scripts are self-contained and should be runnable as-is:

```shell
cd training
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

CUDA_VISIBLE_DEVICES={train_devices} [other parameters] python train_mt5_qa_en_AQA+cs_random.py
```

These scripts will train the model on synthetically-constructed few-shot samples from QA dataset.
In the case of `train*info*.py` scripts, the training demonstrations contain the samples of the same question category,
in the case of `train*random*.py` scripts, the demonstrations are picked randomly.

By default, the scripts are configured to run on a single NVIDIA A100 80GB and cca 20GB of RAM.

### Training Few-shot ICLs specialized to new languages / tasks

If you'd like to analogically train an in-context learner  (1) for a new language, or (2) specialized to a certain task,
we propose to mix the QA dataset (plus other datasets, for (2)), in the training
mix, given the measured cross-lingual transferability, and the demonstrated universality of QA task.

In the case of training in-context learner a new language (1), simply change the values of 
`texts_or_path` (questions), `text_pair_or_path` (contexts) and `labels_or_path` (answers)
in the training scripts.

In the case of specializing the in-context learner to your specific task (2), 
consider keeping the QA objectives for both source and your target language, and adding a new `Priming` objective instance,
with your `texts_or_path` and `labels_or_path`.

Note that these experiments are a part of a larger project aiming to find the most
efficient strategy of picking training demonstrations, 
which is the reason why the training scripts contain some additional complexity,
not necessary for our use-case.