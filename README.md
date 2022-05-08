# XAlign: Cross-lingual-Fact-to-Text-Alignment-and-Generation-for-Low-Resource-Languages

In this work, we propose the creation of cross-lingual fact-to-text dataset, `XAlign` accepted at WebConf-2022 poster-demo track. It consist of English WikiData triples/facts mapped to sentences from low resources Wikipedia. 

We explored two different unsupervised methods to solve cross-lingual alignment task based on: 

- Transfer learning from NLI
- Distant supervision from another English-only dataset

We introduce a large collection of high quality XF2T dataset in 7 languages: Hindi, Marathi, Gujarati, Telugu, Tamil, Kannada, Bengali, and monolingual dataset in English.

This repository consists of steps for executing the cross-lingual alignment approaches and finetuning mT5 for data-to-text generation on XAlign. One can find more details, analyses, and baseline results in [our paper](https://arxiv.org/abs/2202.00291).

## Installation
Install the required packgaes as follow:
```
pip install -r requirements.txt
```

## Dataset
This dataset is available upon request, and one can get the dataset by sharing a short description of how you will use the data and your affiliation to Tushar Abhishek (tushar.abhishek@research.iiit.ac.in).


### Gold standard Test dataset
We manually annotated the test dataset across 8 languages with the help of crowd-sourced annotators.

| Language | #Count | #Word count (avg/min/max) | #Facts/sentence (avg/min/max)
| --- | --- | --- | --- |
|Hindi|842|11.1/5/24|2.1/1/5
Marathi|736|12.7/6/40|2.1/1/8
Telugu|734|9.7/5/30|2.2/1/6
Tamil|656|9.5/5/24|1.9/1/8
English|470|17.5/8/61|2.7/1/7
Gujarati|530|12.7/6/31|2.1/1/6
Bengali|792|8.7/5/24|1.6/1/5
Kannada|642|10.4/6/45|2.2/1/7 


### Train and validation dataset (automatically aligned)
We have automatically created a large collection of well aligned sentence-fact pair across languages using the best cross-lingual aligner evaluated on gold standard test datasets. 

| Language | #Count | #Word Count (avg/min/max) | #Facts/sentence (avg/min/max) |
| --- | --- | --- | --- |
Hindi|56582|25.3/5/99|2.0/1/10|
Marathi|19408|20.4/5/94|2.2/1/10|
Telugu|24344|15.6/5/97|1.7/1/10|
Tamil|56707|16.7/5/97|1.8/1/10|
English|132584|20.2/4/86|2.2/1/10|
Gujarati|9031|23.4/5/99|1.8/1/10|
Bengali|121216|19.3/5/99|2.0/1/10|
Kannada|25441|19.3/5/99|1.9/1/10|

## Cross-lingual Alignment Approaches

### 1) Transfer learning from NLI
Before executing the code, download the XNLI dataset from [here](https://drive.google.com/file/d/1deYzB2NZrELrZwIJhXbY8DdWi0xH__MC/view).

To execute the `mT5` based approach, follow the steps:
```
$ cd XNLI-based-models/finetune_mt5
```
Copy xnli_dataset.zip (downloaded before) to `./datasets` and unzip. Finally execute the command:
```
$ python main.py --epochs 5 --gpus 1 --batch_size 16 --max_seq_len 200 --learning_rate 1e-3 --model_name google/mt5-large --fp16 0
```

To execute the `MuRIL` or `XLM-RoBERTa` based approaches, follow the steps:
```
$ cd XNLI-based-models/finetune_multilingual_encoder_models
```
Copy xnli_dataset.zip (downloaded before) to `./datasets` and unzip.Finally execute the command:
```
$ python main.py --epochs 5 --gpus 1 --batch_size 32 --max_seq_len 200 --learning_rate <lr> --model_name <model_name> --fp16 1
```

where,
- `model_name` can be 'google/muril-large-cased' or 'xlm-roberta-large'
- `learning_rate` must be '1e-5' for 'xlm-roberta-large' or '2e-5' for 'google/muril-large-cased'

### 2) Distant supervision using KELM dataset
Before executing the code, download the multi-lingual KELM dataset from [here](https://drive.google.com/file/d/19h60l-DW5ldS6d4tcr8wm_kKuHl_M2hA/view).

To execute the `mT5` based approach, follow the steps:
```
$ cd distant_supervision/finetune_mt5
```
Copy multilingual-KELM-dataset.zip (downloaded before) to `./datasets` directory and unzip. Finally execute the command:
```
$ python main.py --epochs 5 --gpus 1 --batch_size 16 --max_seq_len 200 --learning_rate 1e-3 --model_name google/mt5-large --fp16 0
```

To execute the `MuRIL` or `XLM-RoBERTa` based approaches, follow the steps:
```
$ cd distant_supervision/finetune_multilingual_encoder_models
```
Copy multilingual-KELM-dataset.zip (downloaded before) to `./datasets` directory and unzip. Finally execute the command:
```
$ python main.py --epochs 5 --gpus 1 --batch_size 32 --max_seq_len 200 --learning_rate <lr> --model_name <model_name> --fp16 1
```

where,
- `model_name` can be 'google/muril-large-cased' or 'xlm-roberta-large'
- `learning_rate` must be '1e-5' for 'xlm-roberta-large' or '2e-5' for 'google/muril-large-cased'

## Cross-lingual Alignment Results
Following are the results for cross-lingual alignment over gold standard test datasets.
|| | | | | F1-score | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| | Hindi | Marathi | Telugu| Tamil | English | Gujarati | Bengali | Kannada | Average |
| **Baselines** | | | | | | | | | |
|[KELM-style](https://aclanthology.org/2021.naacl-main.278/) |49.3|42.6|36.8|45.1|41.0|37.2|43.6|33.8|41.1
|[WITA-style](https://aclanthology.org/2020.emnlp-main.738/) |50.7|57.4|51.7|45.9|60.2|50.0|53.5|53.0|52.8
|Stage-1 + TF-IDF |75.0|68.5|69.3|71.8|73.7|70.1|78.7|64.7|**71.5**
|**Distant Supervision based approaches**|||||||||
MuRIL-large|76.3|68.4|74.0|75.5|70.5|78.5|62.4|67.7|71.7
XLM-Roberta-large|78.1|69.0|76.5|73.9|76.5|78.5|66.9|72.4|74.0
mT5-large|79.0|71.4|77.6|78.6|76.6|80.0|69.8|70.5|**75.4**
|**Transfer Learning based approaches**|||||||||
MuRIL-large|71.6|71.7|76.5|75.1|73.4|78.7|79.5|71.8|74.8
XLM-Roberta-large|77.2|76.7|78.0|81.2|79.0|80.5|83.1|72.7|78.6
mT5-large|90.2|83.1|84.1|88.6|84.5|85.1|75.1|78.5|**83.7**


## Cross-lingual Data-to-Text Generation
Before procedding, copy **XAlign-dataset.zip** (available upon request) to `data-to-text-generator/mT5-baseline/datasets` folder and unzip.

To finetune the best baseline on the XAlign, follow the steps:

```
$ cd data-to-text-generator/mT5-baseline
$ python main.py --epochs 30 --gpus 1 --batch_size 2 --src_max_seq_len 250 --tgt_max_seq_len 200 --learning_rate 1e-3 --model_name google/mt5-small --online_mode 0 --use_pretrained 1 --lang hi,mr,te,ta,en,gu,bn,kn --verbose --enable_script_unification 1 
```

To evaluate the trained model, follow the steps:
```
$ cd data-to-text-generator/mT5-baseline
$ python main.py --epochs 30 --gpus 1 --batch_size 4 --src_max_seq_len 250 --tgt_max_seq_len 200 --learning_rate 1e-3 --model_name google/mt5-small --online_mode 0 --use_pretrained 1 --lang hi,mr,te,ta,en,gu,bn,kn enable_script_unification 1 --inference
```

## Cross-lingual Data-to-Text Generation Results
BLEU obtained on the Test dataset on XAlign.

||Hindi|Marathi|Telugu|Tamil|English|Gujarati|Bengali|Kannada|Average
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Baseline (fact translation)|2.71|2.04|0.95|1.68|1.01|0.64|2.73|0.45|1.53
GAT-Transformer|29.54|17.94|4.91|7.19|40.33|11.34|30.15|5.08|18.31
Vanilla Transformer |35.42|17.31|6.94|8.82|38.87|13.21|35.61|3.16|19.92
mT5-small |40.61|20.23|11.39|13.61|43.65|16.61|45.28|8.77|**25.02**

## Citation
One can cite our [paper](https://arxiv.org/abs/2202.00291) as follows:

```
@article{abhishek2022xalign,
  title={XAlign: Cross-lingual Fact-to-Text Alignment and Generation for Low-Resource Languages},
  author={Abhishek, Tushar and Sagare, Shivprasad and Singh, Bhavyajeet and Sharma, Anubhav and Gupta, Manish and Varma, Vasudeva},
  journal={arXiv preprint arXiv:2202.00291},
  year={2022}
}
```