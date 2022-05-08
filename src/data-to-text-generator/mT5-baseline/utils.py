from collections import defaultdict
import random
import json
import unidecode
from indicnlp.normalize import indic_normalize
from indicnlp.transliterate import unicode_transliterate
from indicnlp.tokenize import indic_tokenize
from sacremoses import MosesPunctNormalizer
from sacremoses import MosesTokenizer
import urduhack
import os


languages_map = {
    'en': {"label": "English", 'id': 0},
    'hi': {"label": "Hindi", 'id': 1},
    'te': {"label": "Telugu", 'id': 2}, 
    'bn': {"label": "Bengali", 'id': 3},
    'pa': {"label": "Punjabi", 'id': 4},
    'ur': {"label": "Urdu", 'id': 5}, 
    'or': {"label": "Odia", 'id': 6}, 
    'as': {"label": "Assamese", 'id': 7},
    'gu': {"label": "Gujarati", 'id': 8},
    'mr': {"label": "Marathi", 'id': 9},
    'kn': {"label": "Kannada", 'id': 10},
    'ta': {"label": "Tamil", 'id': 11},
    'ml': {"label": "Malayalam", 'id': 12} 
}

def handle_multiple_languages(lang):
    # check if multiple "," separated entries exists.
    lang_list = [x.strip().lower() for x in lang.split(',')]
    valid_languages = []
    for x in lang_list:
        if x not in languages_map:
            continue
        valid_languages.append(x)
    if len(valid_languages)!=len(lang_list):
        print("%d invalid languages identified" % (len(lang_list) - len(valid_languages)))
    print('successfully identified %d langauges: %s' % (len(valid_languages), valid_languages))
    valid_languages = sorted(valid_languages)
    return valid_languages

def get_language_normalizer():
    lang_normalizer = {}
    for k in languages_map:
        if k=='ur':
            lang_normalizer[k] = urduhack.normalization
        elif k=='en':
            lang_normalizer[k] = MosesPunctNormalizer()
        else:
            normfactory = indic_normalize.IndicNormalizerFactory()
            lang_normalizer[k] = normfactory.get_normalizer(k)
    return lang_normalizer

def get_id_to_lang():
    id_to_lang = {}
    for k, v in languages_map.items():
        id_to_lang[v['id']] = k
    return id_to_lang

def load_jsonl(file_path):
    res = []
    with open(file_path, encoding='utf-8') as dfile:
        for line in dfile.readlines():
            res.append(json.loads(line.strip()))
    return res

def dataset_exists(dir_path, required_files=['train.jsonl', 'test.jsonl', 'val.jsonl']):
    required_files = set(required_files)
    existing_files = set()
    for dfile in os.listdir(os.path.abspath(dir_path)):
        if dfile in required_files:
            existing_files.add(dfile)
    missing_files = required_files.difference(existing_files)
    if len(missing_files):
        print("%s files are missing, creating the files.." % missing_files)
    return len(missing_files)==0

def store_jsonl(res, file_path):
    with open(file_path, 'w', encoding='utf-8') as dfile:
        for item in res:
            json.dump(item, dfile, ensure_ascii=False)
            dfile.write('\n')

def merge_dataset_across_languages(dataset_dir, languages, dataset_type, merged_file):
    final_res = []
    for lang in languages:
        file_path = os.path.join(dataset_dir, lang, "%s.jsonl" % dataset_type)
        final_res.extend(load_jsonl(file_path))
    random.seed(42)
    random.shuffle(final_res)
    store_jsonl(final_res, merged_file)

def store_txt(res, file_path):
    with open(file_path, 'w', encoding='utf-8') as dfile: 
        for item in res:
            dfile.write("%s\n" % item.strip())

def get_nodes(n):
    n = n.strip()
    n = n.replace('(', '')
    n = n.replace('\"', '')
    n = n.replace(')', '')
    n = n.replace(',', ' ')
    n = n.replace('_', ' ')
    n = unidecode.unidecode(n)
    return n


def get_relation(n):
    n = n.replace('(', '')
    n = n.replace(')', '')
    n = n.strip()
    n = n.split()
    n = "_".join(n)
    return n


def linear_fact_str(fact, enable_qualifiers=False):
    fact_str = ['<R>', get_relation(fact[0]).lower(), '<T>', get_nodes(fact[1]).lower()]
    qualifier_str = [' '.join(['<QR>', get_relation(x[0]).lower(), '<QT>', get_nodes(x[1]).lower()]) for x in fact[2]]
    if enable_qualifiers and len(fact[2])>0:
        fact_str += [' '.join(qualifier_str)]
    return fact_str

def get_text_in_unified_script(text, normalizer, lang):
    return unicode_transliterate.UnicodeIndicTransliterator.transliterate(
                " ".join(
                    indic_tokenize.trivial_tokenize(
                        normalizer.normalize(text.strip()), lang
                    )
                ),
                lang,
                "hi",
            ).replace(" ् ", "्")

def get_native_text_from_unified_script(unified_text, lang):
    return unicode_transliterate.UnicodeIndicTransliterator.transliterate(unified_text, "hi", lang)