import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
from utils import load_jsonl, linear_fact_str, languages_map, get_language_normalizer, get_text_in_unified_script, get_relation
import random
from indicnlp.tokenize import indic_tokenize
from sacremoses import MosesTokenizer
import torch
en_tok = MosesTokenizer(lang="en")


class TextDataset(Dataset):
    def __init__(self, prefix, tokenizer, filename, dataset_count, src_max_seq_len, tgt_max_seq_len, script_unification, logger, complete_coverage, sorted_order=True):
        self.tokenizer = tokenizer
        self.src_max_seq_len = src_max_seq_len
        self.tgt_max_seq_len = tgt_max_seq_len
        self.dataset = load_jsonl(filename)
        if complete_coverage:
            self.dataset = [x for x in self.dataset if x['complete_coverage']==1]
        self.logger = logger
        self.prefix = prefix
        self.sorted_order = sorted_order
        self.script_unification = script_unification
        self.lang_normalizer = get_language_normalizer()
        if dataset_count>0:
            # retain selected dataset count
            self.dataset = self.dataset[:dataset_count]
        data_type = os.path.basename(filename).split('.')[0]
        if self.script_unification:
            logger.critical("%s : script unification to Devanagari is enabled." % data_type)
        logger.info("%s dataset count : %d" % (data_type, len(self.dataset)))
    
    def process_facts(self, facts):
        """ linearizes the facts on the encoder side """
        if self.sorted_order:
            facts = sorted(facts, key=lambda x: get_relation(x[0]).lower())
        linearized_facts = []
        for i in range(len(facts)):
            linearized_facts += linear_fact_str(facts[i], enable_qualifiers=True)
        processed_facts_str = ' '.join(linearized_facts)
        return processed_facts_str

    def process_text(self, text, lang):
        """ normalize and tokenize and then space join the text """
        if lang == 'en':
            return " ".join(en_tok.tokenize(self.lang_normalizer[lang].normalize(text.strip()), escape=False)).strip()
        else:
            # return unified script text
            if self.script_unification:
                return get_text_in_unified_script(text, self.lang_normalizer[lang], lang)
            
            # return original text
            return " ".join(
                indic_tokenize.trivial_tokenize(self.lang_normalizer[lang].normalize(text.strip()), lang)
            ).strip()

    def preprocess(self, text, max_seq_len):
        tokenzier_args = {'text': text, 'truncation': True, 'pad_to_max_length': False, 
                                    'max_length': max_seq_len, 'return_attention_mask': True}
        tokenized_data = self.tokenizer.encode_plus(**tokenzier_args)
        return tokenized_data['input_ids'], tokenized_data['attention_mask']

    def __getitem__(self, idx):
        prefix_str = ''
        data_instance = self.dataset[idx]
        lang_iso = data_instance['lang'].strip().lower()
        lang_id = languages_map[lang_iso]['id']
        if self.prefix:
            prefix_str = "generate  %s : " % languages_map[lang_iso]['label'].lower()
        # preparing the input
        section_info = data_instance['native_sentence_section'] if lang_iso=='en' else data_instance['translated_sentence_section'] 
        input_str = "{prefix}<H> {entity} {triples} <S> {section}".format(prefix=prefix_str, 
                                        entity=data_instance['entity_name'].lower().strip(), triples=self.process_facts(data_instance['facts']),
                                        section=section_info.lower().strip())

        src_ids, src_mask = self.preprocess(input_str, self.src_max_seq_len)
        tgt_ids, tgt_mask = self.preprocess(self.process_text(data_instance['sentence'], lang_iso), self.tgt_max_seq_len)
        return src_ids, src_mask, tgt_ids, tgt_mask, lang_id, idx

    def __len__(self):
        return len(self.dataset)

def pad_seq(seq, max_batch_len, pad_value):
    return seq + (max_batch_len - len(seq)) * [pad_value]

def collate_batch(batch, tokenizer):
    batch_src_inputs = []
    batch_src_masks = []
    batch_tgt_inputs = []
    batch_tgt_masks = []
    lang_id = []
    idx = []

    max_src_len = max([len(ex[0]) for ex in batch])
    max_tgt_len = max([len(ex[2]) for ex in batch])
    
    for item in batch:
        batch_src_inputs += [pad_seq(item[0], max_src_len, tokenizer.pad_token_id)]
        batch_src_masks += [pad_seq(item[1], max_src_len, 0)]
        batch_tgt_inputs += [pad_seq(item[2], max_tgt_len, tokenizer.pad_token_id)]
        batch_tgt_masks += [pad_seq(item[3], max_tgt_len, 0)]
        lang_id.append(item[4])
        idx.append(item[5])

    return torch.tensor(batch_src_inputs, dtype=torch.long), torch.tensor(batch_src_masks, dtype=torch.long), torch.tensor(batch_tgt_inputs, dtype=torch.long), torch.tensor(batch_tgt_masks, dtype=torch.long), torch.tensor(lang_id, dtype=torch.long), torch.tensor(idx, dtype=torch.long)

def get_dataset_loaders(tokenizer, filename, logger, prefix=False, dataset_count=0, batch_size=8, num_threads=0, src_max_seq_len=200, tgt_max_seq_len=200, script_unification=False, complete_coverage=False):
    dataset = TextDataset(prefix, tokenizer, filename, dataset_count, src_max_seq_len, tgt_max_seq_len, script_unification, logger, complete_coverage)
    input_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_threads, collate_fn=lambda x : collate_batch(x, tokenizer))
    return input_dataloader
