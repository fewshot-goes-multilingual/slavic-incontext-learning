from typing import Dict

from datasets import load_dataset, Dataset


def get_all_datasets(training: bool) -> Dict[str, Dataset]:

    dataset_templates_cs = {}
    
    # CNEC:
    dataset = load_dataset("fewshot-goes-multilingual/cs_czech-named-entity-corpus_2.0",
                           split="validation" if not training else "train")
    
    # transform dataset into per-entity form
    samples = []
    for item in dataset:
        text, entities = item.values()
        seen_types = set()
        seen_entities = []
        for entity in entities:
            if entity["category_str"] in seen_types:
                # ambiguous entity type -> rollback addition of all samples of this type
                seen_entities = [e for e in seen_entities if e["label_type"] != entity["category_str"]]
            else:
                seen_entities.append({"text": text,
                                      "label_type": entity["category_str"],
                                      "label": entity["content"]})
                seen_types.add(entity["category_str"])
    
        # add all found entities of the current text
        samples.extend(seen_entities)
    
    dataset_flat = Dataset.from_list(samples)
    
    dataset_templates_cs['fewshot-goes-multilingual/cs_czech-named-entity-corpus_2.0'] = dataset_flat
    
    # CSFD:
    dataset = load_dataset("fewshot-goes-multilingual/cs_csfd-movie-reviews",
                           split="validation" if not training else "train")
    dataset = dataset.filter(lambda x: x["rating_int"] != 3)
    dataset = dataset.map(lambda x: {"label": 1 if x["rating_int"] > 3 else 0})
    
    dataset_templates_cs['fewshot-goes-multilingual/cs_csfd-movie-reviews'] = dataset
    
    # FB-comments:
    dataset = load_dataset("fewshot-goes-multilingual/cs_facebook-comments",
                           split="validation" if not training else "train")
    dataset = dataset.map(lambda x: {"label": x["sentiment_int"] + 1})
    
    dataset_templates_cs["fewshot-goes-multilingual/cs_facebook-comments"] = dataset
    
    # MALL-reviews:
    dataset = load_dataset("fewshot-goes-multilingual/cs_mall-product-reviews",
                           split="validation" if not training else "train")
    dataset = dataset.map(lambda x: {"label": x["rating_int"] + 1})
    
    dataset_templates_cs['fewshot-goes-multilingual/cs_mall-product-reviews'] = dataset

    # SQAD:
    dataset = load_dataset("fewshot-goes-multilingual/cs_squad-3.0",
                           split="validation" if not training else "train")
    
    dataset_templates_cs['fewshot-goes-multilingual/cs_squad-3.0'] = dataset
    
    # NLI:
    dataset = load_dataset("ctu-aic/ctkfacts_nli",
                           split="test" if not training else "train")
    
    dataset_templates_cs['fewshot-goes-multilingual/cs_ctkfacts_nli'] = dataset
    
    return dataset_templates_cs
    