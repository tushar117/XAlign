import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
from utils import load_jsonl, linear_fact_str, languages_map, get_language_normalizer, get_balanced_data
import torch
from indicnlp.tokenize import indic_tokenize
from sacremoses import MosesTokenizer
import torch
en_tok = MosesTokenizer(lang="en")


class TextDataset(Dataset):
    def __init__(self, tokenizer, filename, dataset_count, max_seq_len, logger):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.dataset = load_jsonl(filename)
        self.logger = logger
        self.lang_normalizer = get_language_normalizer()
        if dataset_count>0:
            # ensures that we get stratified dataset
            self.dataset = get_balanced_data(self.dataset, dataset_count, self.logger)
        self.tgt_labels = {
            0: "▁0",
            1: "▁1"
            }
        data_type = os.path.basename(filename).split('.')[0]
        logger.info("%s dataset count : %d" % (data_type, len(self.dataset)))

    def process_text(self, sentence, facts):
        return "".join(['sentence: ', sentence, " facts : ", self.fact_str(facts).strip()])    

    def fact_str(self, fact):
        return ' | '.join(linear_fact_str(fact))

    def preprocess(self, text, max_seq_len):
        tokenzier_args = {'text': text, 'truncation': True, 'pad_to_max_length': False, 
                                    'max_length': max_seq_len, 'return_attention_mask': True}
        tokenized_data = self.tokenizer.encode_plus(**tokenzier_args)
        return tokenized_data['input_ids'], tokenized_data['attention_mask']

    def __getitem__(self, idx):
        data_instance = self.dataset[idx]
        src_ids, src_mask = self.preprocess(self.process_text(data_instance['sentence'], data_instance['fact']), self.max_seq_len)
        tgt_ids, tgt_mask = self.preprocess("%s"%(self.tgt_labels[data_instance['label']]), 3)
        return src_ids, src_mask, tgt_ids, tgt_mask

    def __len__(self):
        return len(self.dataset)

def pad_seq(seq, max_batch_len, pad_value):
    return seq + (max_batch_len - len(seq)) * [pad_value]

def collate_batch(batch, tokenizer):
    batch_src_inputs = []
    batch_src_masks = []
    batch_tgt_inputs = []
    batch_tgt_masks = []
    
    max_src_len = max([len(ex[0]) for ex in batch])
    max_tgt_len = max([len(ex[2]) for ex in batch])
    
    for item in batch:
        batch_src_inputs += [pad_seq(item[0], max_src_len, tokenizer.pad_token_id)]
        batch_src_masks += [pad_seq(item[1], max_src_len, 0)]
        batch_tgt_inputs += [pad_seq(item[2], max_tgt_len, tokenizer.pad_token_id)]
        batch_tgt_masks += [pad_seq(item[3], max_tgt_len, 0)]
        
    return torch.tensor(batch_src_inputs, dtype=torch.long), torch.tensor(batch_src_masks, dtype=torch.long), torch.tensor(batch_tgt_inputs, dtype=torch.long), torch.tensor(batch_tgt_masks, dtype=torch.long)

def get_dataset_loaders(tokenizer, filename, logger, dataset_count=0, batch_size=8, num_threads=0, max_seq_len=200):
    dataset = TextDataset(tokenizer, filename, dataset_count, max_seq_len, logger)
    input_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_threads, collate_fn=lambda x : collate_batch(x, tokenizer))
    return input_dataloader