import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
from utils import get_balanced_data, load_jsonl


class TextDataset(Dataset):
    def __init__(self, tokenizer, filename, dataset_count, max_seq_len, logger):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.dataset = load_jsonl(filename)
        self.logger = logger
        if dataset_count>0:
            # ensures that we get stratified dataset
            self.dataset = get_balanced_data(self.dataset, dataset_count, self.logger)
        self.tgt_labels = {
            "entailment": "▁0",
            "neutral": "▁1",
            "contradiction": "▁2"
            }
        data_type = os.path.basename(filename).split('.')[0]
        logger.info("%s dataset count : %d" % (data_type, len(self.dataset)))

    def process_nli(self, premise: str, hypothesis: str):
        """ process to required xnli format with task prefix """
        return "".join(['xnli: premise: ', premise, ' hypothesis: ', hypothesis])
    
    def preprocess(self, text, max_seq_len):
        tokenzier_args = {'text': text, 'truncation': True, 'padding': 'max_length', 
                                    'max_length': max_seq_len, 'return_attention_mask': True,
                                    "return_tensors":"pt"}
        tokenized_data = self.tokenizer.encode_plus(**tokenzier_args)
        return tokenized_data['input_ids'][0], tokenized_data['attention_mask'][0]

    def __getitem__(self, idx):
        data_instance = self.dataset[idx]
        src_ids, src_mask = self.preprocess(self.process_nli(data_instance['premise'], data_instance['hypothesis']), self.max_seq_len)
        tgt_ids, tgt_mask = self.preprocess("%s"%(self.tgt_labels[data_instance['label']]), 2)
        return src_ids, src_mask, tgt_ids, tgt_mask

    def __len__(self):
        return len(self.dataset)

def get_dataset_loaders(tokenizer, filename, logger, dataset_count=0, batch_size=8, num_threads=0, max_seq_len=200):
    dataset = TextDataset(tokenizer, filename, dataset_count, max_seq_len, logger)
    input_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_threads)
    return input_dataloader