from typing import Callable, Dict, Any

from datasets import load_dataset, Dataset
from promptsource.templates import DatasetTemplates

political_advertasing_translations = {
    'DEFENSE_AND_SECURITY': "Obronność i bezpieczeństwo",
    'EDUCATION': "Edukacja",
    'FOREIGN_POLICY': "Polityka zagraniczna",
    'HEALHCARE': "Służba zdrowia",
    'IMMIGRATION': "Imigracja",
    'INFRASTRUCTURE_AND_ENVIROMENT': "Infrastruktura i środowisko",
    'POLITICAL_AND_LEGAL_SYSTEM': "System polityczny i prawny",
    'SOCIETY': "Społeczeństwo",
    'WELFARE': "Dobrobyt",
}

kpwr_ner_translations = {
    'nam_adj': 'przymiotnik od nazwy własnej',
    'nam_adj_city': 'przymiotnik od nazwy własnej miasta',
    'nam_adj_country': 'przymiotnik od nazwy własnej kraju',
    'nam_adj_person': 'przymiotnik od nazwy własnej osoby',
    'nam_eve': 'nazwa wydarzenia',
    'nam_eve_human': 'nazwa wydarzenia organizowanego przez ludzi',
    'nam_eve_human_cultural': 'nazwa wydarzenia kulturalnego',
    'nam_eve_human_holiday': 'nazwa święta',
    'nam_eve_human_sport': 'nazwa wydarzenia sportowego',
    'nam_fac': 'nazwa budynku',
    'nam_fac_bridge': 'nazwa mostu',
    'nam_fac_goe': 'nazwa budowli',
    'nam_fac_goe_stop': 'nazwa przystanku',
    'nam_fac_park': 'nazwa parku',
    'nam_fac_road': 'nazwa drogi',
    'nam_fac_square': 'nazwa placu',
    'nam_fac_system': 'nazwa systemu',
    "nam_liv": 'nazwa istoty żywej',
    'nam_liv_animal': 'nazwa zwierzęcia',
    'nam_liv_character': 'nazwa postaci',
    'nam_liv_god': 'nazwa bóstwa',
    'nam_liv_habitant': 'nazwa mieszkańca',
    'nam_liv_person': 'nazwa osoby',
    'nam_loc': 'nazwa lokalizacji',
    'nam_loc_astronomical': 'nazwa obiektu astronomicznego',
    'nam_loc_country_region': 'nazwa regionu kraju',
    'nam_loc_gpe_admin1': 'nazwa jednostki administracyjnej pierwszego stopnia',
    'nam_loc_gpe_admin2': 'nazwa jednostki administracyjnej drugiego stopnia',
    'nam_loc_gpe_admin3': 'nazwa jednostki administracyjnej trzeciego stopnia',
    'nam_loc_gpe_city': 'nazwa miasta',
    'nam_loc_gpe_conurbation': 'nazwa aglomeracji',
    'nam_loc_gpe_country': 'nazwa kraju',
    'nam_loc_gpe_district': 'nazwa dzielnicy',
    'nam_loc_gpe_subdivision': 'nazwa osiedla',
    'nam_loc_historical_region': 'nazwa regionu historycznego',
    'nam_loc_hydronym': 'nazwa obiektu wodnego',
    'nam_loc_hydronym_lake': 'nazwa jeziora',
    'nam_loc_hydronym_ocean': 'nazwa oceanu',
    'nam_loc_hydronym_river': 'nazwa rzeki',
    'nam_loc_hydronym_sea': 'nazwa morza',
    'nam_loc_land': 'nazwa terenu',
    'nam_loc_land_continent': 'nazwa kontynentu',
    'nam_loc_land_island': 'nazwa wyspy',
    'nam_loc_land_mountain': 'nazwa góry',
    'nam_loc_land_peak': 'nazwa szczytu',
    'nam_loc_land_region': 'nazwa regionu',
    'nam_num': 'liczba',
    'nam_num_house': 'numer budynku',
    'nam_num_phone': 'numer telefonu',
    'nam_org': 'nazwa organizacji',
    'nam_org_company': 'nazwa firmy',
    'nam_org_group': 'nazwa grupy',
    'nam_org_group_band': 'nazwa zespołu muzycznego',
    'nam_org_group_team': 'nazwa zespołu sportowego',
    'nam_org_institution': 'nazwa instytucji',
    'nam_org_nation': 'nazwa narodu',
    'nam_org_organization': 'nazwa organizacji',
    'nam_org_organization_sub': 'nazwa elementu organizacji',
    'nam_org_political_party': 'nazwa partii politycznej',
    'nam_oth': 'nazwa inna',
    'nam_oth_currency': 'nazwa waluty',
    'nam_oth_data_format': 'nazwa formatu danych',
    'nam_oth_license': 'nazwa licencji oprogramowania',
    'nam_oth_position': 'nazwa stanowiska',
    'nam_oth_tech': 'nazwa technologii',
    'nam_oth_www': 'nazwa strony internetowej',
    'nam_pro': 'nazwa produktu',
    'nam_pro_award': 'nazwa nagrody',
    'nam_pro_brand': 'nazwa marki',
    'nam_pro_media': 'nazwa medium',
    'nam_pro_media_periodic': 'nazwa czasopisma',
    'nam_pro_media_radio': 'nazwa stacji radiowej',
    'nam_pro_media_tv': 'nazwa stacji telewizyjnej',
    'nam_pro_media_web': 'nazwa strony internetowej',
    'nam_pro_model_car': 'nazwa modelu samochodu',
    'nam_pro_software': 'nazwa programu komputerowego',
    'nam_pro_software_game': 'nazwa gry komputerowej',
    'nam_pro_title': 'tytuł utworu',
    'nam_pro_title_album': 'tytuł albumu muzycznego',
    'nam_pro_title_article': 'tytuł artykułu',
    'nam_pro_title_book': 'tytuł książki',
    'nam_pro_title_document': 'tytuł dokumentu',
    'nam_pro_title_song': 'tytuł piosenki',
    'nam_pro_title_treaty': 'tytuł traktatu',
    'nam_pro_title_tv': 'tytuł programu telewizyjnego',
    'nam_pro_vehicle': 'nazwa pojazdu',
}


def process_ner_dataset(
    dataset: Dataset,
    translations: Dict[str, str],
    output_column: str,
    label_mapping: Callable[[str], str] = None
) -> Dataset:
    names = dataset.info.features[output_column].feature.names
    dataset = dataset.map(lambda x: {"tag_names": [names[i] for i in x[output_column]]})

    samples = []
    for item in dataset:
        text = " ".join(item["tokens"])
        seen_entities = []
        for idx, tag in enumerate(item["tag_names"]):
            if tag != "O":
                category = tag.split("-")[1]

                if label_mapping is not None:
                    category = label_mapping(category)
                position = tag.split("-")[0]
                if position == "B":
                    seen_entities.append({
                        "text": text,
                        "label_type": category,
                        "label": item["tokens"][idx],
                    })
                elif position == "I":
                    seen_entities[-1]["label"] += " " + item["tokens"][idx]
        # remove duplicate categories
        seen_categories = set()
        duplicated_categories = set()

        for entity in seen_entities:
            if entity["label_type"] in seen_categories:
                duplicated_categories.add(entity["label_type"])
            else:
                seen_categories.add(entity["label_type"])
        seen_entities = [e for e in seen_entities if e["label_type"] not in duplicated_categories]

        samples.extend(seen_entities)

    dataset_flat = Dataset.from_list(samples)
    dataset_flat = dataset_flat.map(lambda x: {"label_type_selected": translations[x["label_type"]]})
    return dataset_flat


def get_all_datasets(training: bool) -> Dict[str, Dataset]:
    dataset_templates_pl = {}

    # Sentiment
    dataset = load_dataset("clarin-pl/polemo2-official",
                           split="test" if not training else "train")
    
    dataset_templates_pl["fewshot-goes-multilingual/pl_polemo2-official"] = dataset
    
    # NLI
    mapping = {
        "NEUTRAL": 0,
        "ENTAILMENT": 1,
        "CONTRADICTION": 2
    }
    dataset = load_dataset("allegro/klej-cdsc-e",
                           split="test" if not training else "train")
    dataset = dataset.map(lambda x: {"label": int(mapping[x["entailment_judgment"]])})
    
    dataset_templates_pl["fewshot-goes-multilingual/pl_klej-cdsc-e"] = dataset
    
    # NER - political advertising
    dataset = load_dataset("laugustyniak/political-advertising-pl",
                           split="test" if not training else "train")
    dataset_processed = process_ner_dataset(dataset, political_advertasing_translations, "tags")
    
    dataset_templates_pl["fewshot-goes-multilingual/pl_political-advertising-pl"] = dataset_processed
    
    # NER - generic
    dataset = load_dataset("clarin-pl/kpwr-ner",
                           split="test" if not training else "train")

    # All classes
    dataset_processed = process_ner_dataset(dataset, kpwr_ner_translations, "ner")

    # Only generic classes
    dataset_processed = process_ner_dataset(dataset, kpwr_ner_translations, "ner", lambda x: "_".join(x.split("_")[0:2]))
    
    dataset_templates_pl["fewshot-goes-multilingual/pl_kpwr-ner"] = dataset_processed

    return dataset_templates_pl
