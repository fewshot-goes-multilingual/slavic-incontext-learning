
# In-context Few-shot Learners for Slavic Languages

This repository contains resources for in-context learning in Slavic languages, 
together with training scripts for creating simple in-context few-shot learners specialized to an arbitrary target language.

This is a supplementary material to an article [Resources and Few-shot Learners
for In-context Learning in Slavic Languages](TODO) presented on EACL SlavNLP 2023.
For more details, take a look into the referenced paper.

### Overview

To enable evaluating and training in-context learners in **Polish, Russian and Czech**, we deliver the following:

1. **survey the existing datasets** for diverse tasks in these languages, 
2.  **transform** them to in-context learning format through **templates** that we collect 
and proofread by the native speakers of datasets' language
3. **evaluate** the existing SOTA in-context learners in our target languages
4. **train** smaller, yet for many applications superior in-context few-shot learners for Czech and Russian.

Following sections show how to use our collected templates and trained in-context learners.

See [Training](training) section to reproduce our training method for other languages, 
or [Evaluation](evaluation) section on how to reproduce the reported results, or evaluate your own in-context learner for Czech, Polish, or Russian. 

## Using new prompt / instruction templates

We collect new templates within the BigScience's [Promptsource platform](https://github.com/bigscience-workshop/promptsource)
and thus, the templates can be used as any other promptsource template:

Install HuggingFace Datasets and our [Promptsource fork](https://github.com/fewshot-goes-multilingual/promptsource):
```shell
pip install datasets git+https://github.com/fewshot-goes-multilingual/promptsource.git
```
Verbalize a dataset sample:
```python
from datasets import load_dataset
from promptsource.templates import DatasetTemplates

dataset_id = "ctu-aic/ctkfacts_nli"  # ID in HuggingFace datasets
templates_id = "fewshot-goes-multilingual/cs_ctkfacts_nli"  # ID of the promptsource templates

dataset = load_dataset(dataset_id, split="train")
dataset_templates = DatasetTemplates(templates_id)

dataset[0]
# >>> {'id': 1306, 'label': 2, 'evidence': 'PRAHA 18. června (ČTK) - Rekordní teploty 19. června (od roku 1775 měřené v pražském Klementinu) byly následující: nejvyšší teplota 31,2 z roku 1917 a 1934, nejnižší teplota 7,3 z roku 1985\\. Dlouhodobý průměrný normál: 17,9 stupně Celsia.', 'claim': 'Rekordní teploty se od roku 1775 měří v Praze.'}
dataset_templates.all_template_names
# >>> ['GPT-3 style', 'MNLI crowdsource', 'based on the previous passage', 'does it follow that', 'must be true', 'should assume', 'take the following as truth']

# note that before , some datasets (such as NER) require transformations
# you can find these transformations in evaluation/tasks/{lang}/customized_datasets.py
prompt, label = dataset_templates['GPT-3 style'].apply(dataset[0])

prompt
# >>> 'PRAHA 18. června (ČTK) - Rekordní teploty 19. června (od roku 1775 měřené v pražském Klementinu) byly následující: nejvyšší teplota 31,2 z roku 1917 a 1934, nejnižší teplota 7,3 z roku 1985\\. Dlouhodobý průměrný normál: 17,9 stupně Celsia.\nOtázka: Rekordní teploty se od roku 1775 měří v Praze. Pravda, nepravda, nebo ani jedno?'
label
# >>> 'Pravda'
```


### Datasets index

Following is a list of (Task_type) `dataset` -> `templates` that we make available. 
Each `templates` contains 3-7 different templates per dataset.

* Czech:
  * (NER) `fewshot-goes-multilingual/cs_czech-named-entity-corpus_2.0` -> `fewshot-goes-multilingual/cs_czech-named-entity-corpus_2.0`
  * (Sentiment) `fewshot-goes-multilingual/cs_csfd-movie-reviews` -> `fewshot-goes-multilingual/cs_csfd-movie-reviews`
  * (Sentiment) `fewshot-goes-multilingual/cs_facebook-comments` -> `fewshot-goes-multilingual/cs_facebook-comments`
  * (Sentiment) `fewshot-goes-multilingual/cs_mall-product-reviews` -> `fewshot-goes-multilingual/cs_mall-product-reviews`
  * (QA) `fewshot-goes-multilingual/cs_squad-3.0` -> `fewshot-goes-multilingual/cs_squad-3.0`
  * (NLI) `ctu-aic/ctkfacts_nli` -> `fewshot-goes-multilingual/cs_squad-3.0`
* Polish:
  * (NER) `laugustyniak/political-advertising-pl` -> `fewshot-goes-multilingual/pl_political-advertising-pl`
  * (NER) `clarin-pl/kpwr-ner` -> `fewshot-goes-multilingual/pl_kpwr-ner`
  * (Sentiment) `clarin-pl/polemo2-official` -> `fewshot-goes-multilingual/pl_polemo2-official`
  * (NLI) `allegro/klej-cdsc-e` -> `fewshot-goes-multilingual/pl_klej-cdsc-e`
* Russian:
  * (NER) `polyglot_ner` -> `fewshot-goes-multilingual/ru_polyglot_ner`
  * (Sentiment) `cedr` -> `fewshot-goes-multilingual/ru_cedr`
  * (QA) `sberquad` -> `fewshot-goes-multilingual/ru_sberquad`
  * (NLI) `xnli` -> `fewshot-goes-multilingual/ru_xnli`

Note that for your convenience, we make some of the datasets newly available 
on [HuggingFace Datasets](https://huggingface.co/datasets), we always reference the datasets creators 
and their licensing conditions.
Please don't forget to give credit to the datasets' authors if you use their dataset.

Soon we plan to merge the newly-crafted templates to the upstream Promptsource repository in the future.

## Using newly-trained Few-shot In-context Learners

The best-performing in-context learners that we newly create can be found among our [HuggingFace models](https://huggingface.co/fewshot-goes-multilingual).
See the models cards ([Czech model card](https://huggingface.co/fewshot-goes-multilingual/mTk-SQuAD_en-SQAD_cs-1B) & [Russian model card](https://huggingface.co/fewshot-goes-multilingual/mTk-AdversarialQA_en-SberQuAD_ru-1B)) for the specifics of the prompting format, and
[corresponding paper](TODO) for performance evaluation.

The models can be loaded and prompted as follows:
```bash
pip install transformers
```

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("fewshot-goes-multilingual/mTk-SQuAD_en-SQAD_cs-1B")
tokenizer = AutoTokenizer.from_pretrained("fewshot-goes-multilingual/mTk-SQuAD_en-SQAD_cs-1B")

# construct demonstrations e.g. from dataset verbalized in the example above 
fewshot_input_outputs = [dataset_templates['GPT-3 style'].apply(dataset[i]) for i in range(3)]
predicted_input, true_label = dataset_templates['GPT-3 style'].apply(dataset[3])

# construct the full few-shot prompt for predicting `predicted_input` label
fewshot_prompt = "\n".join("Input: %s Prediction: %s" % (prompt, label) for prompt, label in fewshot_input_outputs)
fewshot_prompt += "\nInput: %s"
fewshot_prompt += predicted_input
fewshot_prompt += " Prediction: "

encoded_input = tokenizer(fewshot_prompt, return_tensors="pt")
model_output = model.generate(**encoded_input)
predicted_text = tokenizer.batch_decode(model_output, skip_special_tokens=True)[0]

predicted_text
# >>> 'nepravda'
predicted_text == true_label
# >>> True
```

## Citation

If you use our models or resources in your research, please cite this work as follows.

### Text

ŠTEFÁNIK, Michal, Marek KADLČÍK, Piotr GRAMACKI and Petr SOJKA. Resources and Few-shot Learners for In-context Learning in Slavic Languages. In *Proceedings of the 9th Workshop on Slavic Natural Language Processing*. ACL, 2023. 9 pp.

### BibTeX

```bib
@inproceedings{stefanik2023resources,
  author = {\v{S}tef\'{a}nik, Michal and Kadlčík, Marek and Gramacki, Piotr and Sojka, Petr},
  title = {Resources and Few-shot Learners for In-context Learning in Slavic Languages},
  booktitle = {Proceedings of the 9th Workshop on Slavic Natural Language Processing},
  publisher = {ACL},
  numpages = {9},
  url = {TO BE FILLED},
}
```

