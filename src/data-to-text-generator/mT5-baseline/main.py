import os,sys, re
import random
import json
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from utils import (languages_map, get_language_normalizer, 
        get_id_to_lang, store_txt, get_native_text_from_unified_script, 
        handle_multiple_languages, merge_dataset_across_languages, dataset_exists)
from indicnlp.tokenize import indic_tokenize
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from collections import defaultdict


from transformers import (
    AdamW,
    Adafactor,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup
)

from logger import MyLogger, LOG_LEVELS
from dataloader import get_dataset_loaders
from sacrebleu.metrics import BLEU


base_dir = os.path.dirname(os.path.realpath(__file__))
from sacremoses import MosesTokenizer
en_tok = MosesTokenizer(lang="en")

# allow deterministic psuedo-random-initialization
def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ModelWrapper(pl.LightningModule):
    def __init__(self, args):
        super(ModelWrapper, self).__init__()
        self.step_loss_labels = {'train': 'loss', 'val': 'val_loss', 'test': 'test_loss'}
        self.config_args = args
        self.tokenizer = MT5Tokenizer.from_pretrained(args.model_name, cache_dir="/tmp/hugginface")
        self.add_special_tokens()
        self.cal_bleu = BLEU(tokenize='none')
        self.lang_normalizer = get_language_normalizer()

        if args.use_pretrained:
            # using pretrained transformers
            self.model = MT5ForConditionalGeneration.from_pretrained(args.model_name, dropout_rate=args.dropout_rate, cache_dir="/tmp/hugginface")
        else:
            # training transformer from scratch
            self.model = MT5ForConditionalGeneration.from_config(MT5ForConditionalGeneration.from_pretrained(
                args.model_name, dropout_rate=args.dropout_rate, cache_dir="/tmp/hugginface"))
        self.model.resize_token_embeddings(len(self.tokenizer))

    def add_special_tokens(self):
        new_tokens = ['<H>', '<R>', '<T>', '<QR>', '<QT>', '<S>']
        new_tokens_vocab = {}
        new_tokens_vocab['additional_special_tokens'] = []
        for idx, t in enumerate(new_tokens):
            new_tokens_vocab['additional_special_tokens'].append(t)
        num_added_toks = self.tokenizer.add_special_tokens(new_tokens_vocab)
        self.config_args.logger.critical('added %s tokens' % num_added_toks)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):    
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
            decoder_attention_mask=decoder_attention_mask
        )
        return outputs
        
    def configure_optimizers(self):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config_args.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = Adafactor(optimizer_grouped_parameters, lr=self.config_args.learning_rate, scale_parameter=False, relative_step=False, warmup_init=False)
        # optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.config_args.learning_rate, eps=1e-6)

        if self.config_args.enable_scheduler:
            total_dataset_count = self.config_args.train_dataset_count
            total_steps = int(np.ceil((self.config_args.epochs * total_dataset_count) /
                              (self.config_args.batch_size*self.config_args.gpus)))

            scheduler = {
                # 'scheduler': get_constant_schedule_with_warmup(optimizer, self.config_args.warmup_steps*total_steps)
                'scheduler': get_linear_schedule_with_warmup(optimizer, self.config_args.warmup_steps*total_steps, total_steps),
                'interval': 'step',
            }
            return [optimizer], [scheduler]

        return optimizer

    def ids_to_clean_text(self, generated_ids, remove_special_tokens=True, remove_tok_spaces=True):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=remove_special_tokens, clean_up_tokenization_spaces=remove_tok_spaces
        )
        return list(map(str.strip, gen_text))

    def _generative_step(self, batch):
        generated_ids = self.model.generate(
            batch[0],
            attention_mask=batch[1],
            use_cache=True,
            num_beams=self.config_args.eval_beams,
            max_length=self.config_args.tgt_max_seq_len,
            length_penalty=self.config_args.length_penalty,
        )

        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(batch[2])
        return {'predicted_txt': preds, 'target_txt':target}

    def _recover_input_text(self, input_ids):
        input_txt = self.ids_to_clean_text(input_ids, remove_special_tokens=False)
        processed_input_ids = []
        for i in range(len(input_txt)):
            final_str = input_txt[i]
            # removing model's default special token
            for special_token in [self.tokenizer.eos_token, self.tokenizer.pad_token]:
                if not special_token:
                    continue
                final_str = re.sub(special_token, '', final_str)
            input_txt[i] = final_str.strip()
        return input_txt

    def _step(self, batch, step_type):
        lm_labels = torch.clone(batch[2])
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            batch[0],
            attention_mask=batch[1],
            lm_labels=lm_labels,
            decoder_attention_mask=batch[3]
        )
        return_map = {}
        online_logger_data = {}
        return_map.update({self.step_loss_labels[step_type]:outputs[0]}) 
        # updating the online logger
        online_logger_data.update(return_map)
        # inserting the predictions and actual text for validation and testing dataset when executing bi-lingual training
        # inserting the predictions and actual text for test dataset only when executing multi-lingual training
        if step_type!='train' and ((len(self.config_args.lang)==1 and self.config_args.enable_bleu_cal_per_epoch==1) or step_type=='test'):
            with torch.no_grad():
                return_map.update(self._generative_step(batch))
                return_map.update({'source_txt': self._recover_input_text(batch[0])})
                return_map.update({'lang_id': batch[4].cpu().detach().tolist()})
                return_map.update({'seq_id': batch[5].cpu().detach().tolist()})
        self.logger.log_metrics(online_logger_data)
        return return_map

    def _preprocess_text(self, text, lang):
        native_text = text
        if self.config_args.enable_script_unification>0 and lang!='en':
            # convert unified script to native langauge text
            native_text = get_native_text_from_unified_script(text, lang)

        native_text = native_text.strip()
        # as input and predicted text are already space tokenized
        native_text = ' '.join([x for x in native_text.split()])
        return native_text

    def _epoch_end(self, step_outputs, end_type):
        loss_label = self.step_loss_labels[end_type]
        avg_loss = torch.stack([x[loss_label] for x in step_outputs]).mean()
        if end_type!='train' and ((len(self.config_args.lang)==1 and self.config_args.enable_bleu_cal_per_epoch==1) or end_type=='test'):
            id_to_lang = get_id_to_lang()
            src_txt = []
            pred_txt = []
            ref_txt = []
            lang_info = []
            seq_id = []
            for z in step_outputs:
                src_txt.extend(z['source_txt'])
                pred_txt.extend(z['predicted_txt'])
                ref_txt.extend(z['target_txt'])
                lang_info.extend([id_to_lang[u] for u in z['lang_id']])
                seq_id.extend(z['seq_id'])
            # normalizing and tokenizing using indic_tokenize
            pred_txt = [self._preprocess_text(x, l) for x, l in zip(pred_txt, lang_info)]
            ref_txt = [self._preprocess_text(x, l) for x, l in zip(ref_txt, lang_info)]

            # visual model prediction on wandb
            if self.config_args.verbose and ((len(self.config_args.lang)==1 and self.config_args.enable_bleu_cal_per_epoch==1) or end_type=='test'):
                if self.current_epoch==0 or end_type=='test':
                    # writing the model generated outputs
                    store_txt(src_txt, os.path.join(self.config_args.verbose_output_dir, '%s-src.txt' % end_type))
                    store_txt(ref_txt, os.path.join(self.config_args.verbose_output_dir, '%s-ref.txt' % end_type))
                store_txt(pred_txt, os.path.join(self.config_args.verbose_output_dir, '%s-predicted-epoch-%d.txt' % (end_type, self.current_epoch)))

            # calculating the bleu score
            if (len(self.config_args.lang)==1 and self.config_args.enable_bleu_cal_per_epoch==1):
                # handling bilingual setting
                bleu = self.cal_bleu.corpus_score(pred_txt, [ref_txt])
                self.config_args.logger.info("epoch : %d | %s" % (self.current_epoch, bleu))
                bleu_list = list(map(float, bleu.prec_str.split('/')))
                self.logger.log_metrics({
                                    'overall_%s_bleu' % end_type : bleu.score,
                                    '%s_bleu_1' % end_type : bleu_list[0],
                                    '%s_bleu_2' % end_type : bleu_list[1],
                                    '%s_bleu_3' % end_type : bleu_list[2],
                                    '%s_bleu_4' % end_type : bleu_list[3]
                                    })
                self.log('overall_%s_bleu' % end_type, bleu.score, prog_bar=True)
            else:
                # handling multilingual setting
                language_specific_inout = defaultdict(lambda: {'pred': [], 'ref': []})
                for lpred, lref, llang in zip(pred_txt, ref_txt, lang_info):
                    language_specific_inout[llang]['pred'].append(lpred)
                    language_specific_inout[llang]['ref'].append(lref)
                
                for klang, vdata in language_specific_inout.items():
                    bleu = self.cal_bleu.corpus_score(vdata['pred'], [vdata['ref']])
                    self.config_args.logger.info("%s : epoch : %d | %s" % (klang, self.current_epoch, bleu))
                    bleu_list = list(map(float, bleu.prec_str.split('/')))
                    self.logger.log_metrics({
                                        '%s_overall_%s_bleu' % (klang, end_type) : bleu.score,
                                        '%s_%s_bleu_1' % (klang, end_type) : bleu_list[0],
                                        '%s_%s_bleu_2' % (klang, end_type) : bleu_list[1],
                                        '%s_%s_bleu_3' % (klang, end_type) : bleu_list[2],
                                        '%s_%s_bleu_4' % (klang, end_type) : bleu_list[3]
                                        })
                    self.log('%s_overall_%s_bleu' % (klang, end_type), bleu.score, prog_bar=False)

        self.config_args.logger.info('epoch : %d - average_%s_loss : %f' % (self.current_epoch, end_type, avg_loss.item() ))
        # logging to weight and bias if online mode is enabled
        self.logger.log_metrics(
            {'avg_%s_loss' % end_type: avg_loss})
        self.log('avg_%s_loss' % end_type, avg_loss, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def training_epoch_end(self, train_step_outputs):
        self._epoch_end(train_step_outputs, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def validation_epoch_end(self, val_step_outputs):
        self._epoch_end(val_step_outputs, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._step(batch, 'test')

    def test_epoch_end(self, test_step_outputs):
        self._epoch_end(test_step_outputs, 'test')

    def _intiate_dataset_merging(self, dataset_type, dataset_dir, languages, logger):
        logger.critical('%s: merging the %d %s different languages dataset' % (dataset_type, len(languages), languages))
        final_dir_path = os.path.join(os.path.abspath(dataset_dir), '-'.join(languages))
        os.makedirs(final_dir_path, exist_ok=True)
        # check if dataset already exists
        if dataset_exists(final_dir_path, required_files=['%s.jsonl' % dataset_type]):
            logger.info("%s dataset is already present." % dataset_type)
        else:
            merge_dataset_across_languages(dataset_dir, languages, dataset_type, os.path.join(final_dir_path, "%s.jsonl" % dataset_type))
        return final_dir_path

    def val_dataloader(self):
        script_unification = self.config_args.enable_script_unification > 0
        coverage_flag = self.config_args.complete_coverage > 0
        if len(self.config_args.lang)==1:
            enable_prefix = False
            dev_file_path = os.path.join(os.path.abspath(self.config_args.dataset_path), self.config_args.lang[0], 'val.jsonl')
        else:
            enable_prefix = True
            merged_directory = self._intiate_dataset_merging('val', self.config_args.dataset_path, self.config_args.lang, 
                                                                            self.config_args.logger)
            dev_file_path = os.path.join(os.path.abspath(merged_directory), 'val.jsonl')
        val_dataset = get_dataset_loaders(self.tokenizer, dev_file_path, self.config_args.logger, dataset_count=self.config_args.val_dataset_count, 
                                          batch_size=self.config_args.eval_batch_size, src_max_seq_len=self.config_args.src_max_seq_len, 
                                          tgt_max_seq_len=self.config_args.tgt_max_seq_len, script_unification=script_unification, prefix=enable_prefix,
                                          complete_coverage=coverage_flag)
        return val_dataset

    def train_dataloader(self):
        script_unification = self.config_args.enable_script_unification>0
        coverage_flag = self.config_args.complete_coverage > 0
        if len(self.config_args.lang)==1:
            enable_prefix = False
            train_file_path = os.path.join(os.path.abspath(self.config_args.dataset_path), self.config_args.lang[0], 'train.jsonl')
        else:
            enable_prefix = True
            merged_directory = self._intiate_dataset_merging('train', self.config_args.dataset_path, self.config_args.lang, 
                                                                            self.config_args.logger)
            train_file_path = os.path.join(os.path.abspath(merged_directory), 'train.jsonl')
        train_dataset = get_dataset_loaders(self.tokenizer, train_file_path, self.config_args.logger, dataset_count=self.config_args.train_dataset_count,
                                            batch_size=self.config_args.batch_size,  src_max_seq_len=self.config_args.src_max_seq_len, 
                                            tgt_max_seq_len=self.config_args.tgt_max_seq_len, script_unification=script_unification, prefix=enable_prefix,
                                            complete_coverage=coverage_flag)
        return train_dataset
    
    def test_dataloader(self):
        script_unification = self.config_args.enable_script_unification>0
        if len(self.config_args.lang)==1:
            enable_prefix = False
            test_file_path = os.path.join(os.path.abspath(self.config_args.dataset_path), self.config_args.lang[0], 'test.jsonl')
        else:
            enable_prefix = True
            merged_directory = self._intiate_dataset_merging('test', self.config_args.dataset_path, self.config_args.lang, 
                                                                            self.config_args.logger)
            test_file_path = os.path.join(os.path.abspath(merged_directory), 'test.jsonl')
        test_dataset = get_dataset_loaders(self.tokenizer, test_file_path, self.config_args.logger, dataset_count=self.config_args.test_dataset_count,
                                            batch_size=self.config_args.eval_batch_size,  src_max_seq_len=self.config_args.src_max_seq_len, 
                                            tgt_max_seq_len=self.config_args.tgt_max_seq_len, script_unification=script_unification, prefix=enable_prefix)
        return test_dataset

def get_checkpoint_file(checkpoint_path, logger):
    file_list = []
    for file_name in os.listdir(checkpoint_path):
        if not file_name.endswith('ckpt'):
            continue
        last_modified_time = os.path.getmtime(
            os.path.join(checkpoint_path, file_name))
        file_list.append([file_name, last_modified_time])

    logger.info(
        'total number of files within checkpoint directory [%s]: %d' % (checkpoint_path, len(file_list)))
    if len(file_list) == 0:
        return False, ""
    # if multiple files exists then choose the last modified checkpoint path
    file_list = sorted(file_list, key=lambda x: x[1], reverse=True)
    return True, os.path.join(checkpoint_path, file_list[0][0])

def start_training(args):
    model_name = args.logger_exp_name

    args.logger.debug('initiating training process...')

    if args.inference:
        actual_model_name = model_name.split('-', 1)
        final_checkpoint_path = os.path.join(args.checkpoint_path, actual_model_name[1])
    else:
        final_checkpoint_path = os.path.join(args.checkpoint_path, model_name)
    os.makedirs(final_checkpoint_path, exist_ok=True)
    
    call_back_parameters = {
        'filepath': final_checkpoint_path,
        'save_top_k': 1,
        'verbose': True,
        'monitor': 'overall_val_bleu' if (len(args.lang)==1 and args.enable_bleu_cal_per_epoch==1) else 'avg_val_loss',
        'mode': 'max' if (len(args.lang)==1 and args.enable_bleu_cal_per_epoch==1) else 'min',
    }

    # checkpoint callback to used by the Trainer
    checkpoint_callback = ModelCheckpoint(**call_back_parameters)
    
    # # checkpoint save function for newer version of pytorch lightning  
    # # checkpoint callback to used by the Trainer that saves file like: my/path/epoch=02-val_loss=0.32.ckpt
    # checkpoint_callback = ModelCheckpoint(
    #     monitor="overall_val_bleu",
    #     dirpath=final_checkpoint_path,
    #     filename="{epoch:02d}-{val_bleu:.2f}",
    #     save_top_k=1,
    #     mode="max",
    # )

    # early stop callback
    early_stop_callback = EarlyStopping(
        monitor='overall_val_bleu' if (len(args.lang)==1 and args.enable_bleu_cal_per_epoch==1) else 'avg_val_loss',
        patience=args.patience,
        verbose=True,
        mode='max' if (len(args.lang)==1 and args.enable_bleu_cal_per_epoch==1) else 'min',
    )

    model = ModelWrapper(args)

    args.logger.debug(model)
    args.logger.info('Model has %d trainable parameters' %
                     count_parameters(model))

    callback_list = [checkpoint_callback, early_stop_callback]

    global_callback_params = {
        "callbacks": callback_list,
        "max_epochs": args.epochs,
        "min_epochs": 1,
        "gradient_clip_val": args.clip_grad_norm,
        "gpus": 1 if args.inference else args.gpus,
        "distributed_backend": "ddp",
        "logger": args.online_logger
    }

    #checking whether checkpoint already exists or not
    checkpoint_exists, checkpoint_file = get_checkpoint_file(final_checkpoint_path, args.logger)
    if checkpoint_exists:
        global_callback_params.update({'resume_from_checkpoint': checkpoint_file})
        args.logger.info('resuming training from checkpoint : %s' % checkpoint_file)

    trainer = pl.Trainer(**global_callback_params)
    if args.inference:
        if not checkpoint_exists:
            args.logger.error('No checkpoint found in directory : %s' % final_checkpoint_path)
            sys.exit(0)
        args.logger.debug('about to start testing loop...')
        # change gpus to 1 while testing
        trainer.gpus = 1
        trainer.test(model=model, ckpt_path=checkpoint_file)
        args.logger.debug('testing done.')
    else:
        # finally train the model
        args.logger.debug('about to start training loop...')
        trainer.fit(model)
        args.logger.debug('training done.')
    # args.logger.debug('about to start testing loop...')
    # # change gpus to 1 while testing
    # trainer.gpus = 1
    # trainer.test(ckpt_path="best")
    # args.logger.debug('testing done.')


if __name__ == "__main__":
    parser = ArgumentParser()

    default_checkpoint_path = os.path.join(base_dir, 'lightning_checkpoints')
    default_verbose_output = os.path.join(base_dir, 'model_outputs')

    # Global model configuration
    parser.add_argument('--checkpoint_path', default=default_checkpoint_path, type=str,
                        help='directory where checkpoints are stored')
    parser.add_argument('--dataset_path', required=True, type=str,
                        help='directory where dataset exits')
    parser.add_argument('--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--epochs', default=1, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='adjust batch size per gpu')
    parser.add_argument('--eval_batch_size', default=4, type=int,
                        help='adjust batch size per gpu')
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        help='specify the learning rate')
    parser.add_argument('--clip_grad_norm', default=0.0, type=float,
                        help='clip gradients with norm above specified value, 0 value will disable it.')
    parser.add_argument('--weight_decay', default=0.01, type=float,
                        help='specify the weight decay.')
    parser.add_argument('--dropout_rate', default=0.1, type=float,
                        help='specify the dropout rate for all layer, also applies to all transformer layers.')
    parser.add_argument('--patience', default=10, type=int,
                        help='specify patience for early stop algorithm. if its 0 then disable this feature.')
    # parser.add_argument('--seed', default=42, type=int,
    # help='seed value for random initialization.')
    parser.add_argument("--enable_scheduler", action='store_true',
                        help='activates the linear decay scheduler.')
    parser.add_argument("--warmup_steps", default=4000, type=int,
                        help="percentage of total step used as linear warmup while training the model.")
    # below three arguments are for debugging purpose
    parser.add_argument("--train_dataset_count", type=int, default=0,
                        help="specify number of training data to use. (for debugging purpose). If zero then takes all the available dataset.")
    parser.add_argument("--val_dataset_count", type=int, default=0,
                        help="specify number of validation data to use. (for debugging purpose). If zero then takes all the available dataset.")
    parser.add_argument("--test_dataset_count", type=int, default=0,
                        help="specify number of test data to use. (for debugging purpose). If zero then takes all the available dataset.")
    
    # logger configs
    parser.add_argument('--online_mode', default=0, type=int,
                        help='disables weight and bias syncronization if 0 is passed')
    
    # visualizing the model outputs
    parser.add_argument("--verbose", action='store_true',
                        help="logs model text generation predictions on wandb dashboard.")
    parser.add_argument("--verbose_output_dir", type=str, default=default_verbose_output,
                        help="create a directory for storing source text, reference text and generared text at each epoch (for both testing and validation).") 

    # architecture configs
    parser.add_argument('--model_name', type=str, default='google/mt5-small',
                        help='specify pretrained transformer model to use.')
    parser.add_argument('--use_pretrained', type=int, default=1,
                        help='loads pretrained transformer model.')
    
    # text generation config
    parser.add_argument('--enable_bleu_cal_per_epoch', type=int, default=0,
                        help="specify value greater than 0 to calculate the bleu score per epoch for monolingual setting. warning this may slow down the training process")
    parser.add_argument('--src_max_seq_len', type=int, default=200,
                        help="specify the maximum sequence length for input sequence.")
    parser.add_argument('--tgt_max_seq_len', type=int, default=200,
                        help="specify the maximum sequence length for output sequence.")
    parser.add_argument('--lang', type=str, required=True, 
                        help='specify the target language iso code. Mutliple languages supported if their iso codes are separated using ",".')
    parser.add_argument('--length_penalty', type=float, default=1.0,
                        help='specify the text generation length penalty.')
    parser.add_argument('--eval_beams', type=int, default=5,
                        help='specify size of beam search.')
    # script unification
    parser.add_argument('--enable_script_unification', type=int, default=0,
                        help="specify value greater than 0 to enable script unification to Devanagri for Indic languages.")

    # coverage specifier
    parser.add_argument('--complete_coverage', typ=int, default=0,
                        help="specify whether to use only complete coverage dataset or entire dataset.")

    # inference
    parser.add_argument('--inference', action='store_true',
                        help="enables inference on stored checkpoint")
    args = parser.parse_args()

    args.lang = handle_multiple_languages(args.lang)
    if(len(args.lang)==0):
        print('Invalid language(s) specified !!!')
        sys.exit(0)

    args.logger_exp_name = "%s-%s-%s-%s" % ('-'.join(args.lang), args.model_name, args.epochs, args.learning_rate)
    args.logger_exp_name = args.logger_exp_name.replace('/', '-')

    if args.complete_coverage > 0:
        args.logger_exp_name = "%s-high-coverage" % args.logger_exp_name

    if args.enable_script_unification > 0:
        args.logger_exp_name = "%s-unified-script" % args.logger_exp_name

    if args.inference:
        args.logger_exp_name = "inference-%s" % args.logger_exp_name

    # offline logger
    args.logger = MyLogger('', os.path.join(base_dir, "%s.log" % args.logger_exp_name),
                           use_stdout=True, log_level=LOG_LEVELS.DEBUG, overwrite=True)


    # get the arguments passed to this program
    params = {}
    for arg in vars(args):
        if arg in ["online_logger", "logger"]:
            continue
        params[arg] = getattr(args, arg)

    logger_args = {
        'project': 'multilingual-data-to-text-generation',    # first create a project on weight & bias with local account
        'name': args.logger_exp_name,
        'config': params,
        'tags': ['pytorch-lightning'],
    }

    # turn off the online sync
    if args.online_mode == 0:
        logger_args.update({'offline': True}),

    # configure and add logger to arguments
    args.online_logger = WandbLogger(**logger_args)

    # get the arguments passed to this program
    args.logger.info('\ncommand line argument captured ..')
    args.logger.info('--'*30)

    for key, value in params.items():
        args.logger.info('%s - %s' % (key, value))
    args.logger.info('--'*30)

    if args.verbose:
        # creating table for verbose setting
        args.verbose_output_dir = os.path.join(args.verbose_output_dir, args.logger_exp_name)
        os.makedirs(args.verbose_output_dir, exist_ok=True)
    start_training(args)
