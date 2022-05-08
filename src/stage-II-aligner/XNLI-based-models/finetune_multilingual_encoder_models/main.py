import os,sys
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

from transformers import (
    AdamW,
    
    Adafactor,
    MT5ForConditionalGeneration,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup
)

from logger import MyLogger, LOG_LEVELS
from dataloader import get_dataset_loaders


base_dir = os.path.dirname(os.path.realpath(__file__))


# allow deterministic psuedo-random-initialization
def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TexClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, hidden_size, num_labels, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        x = features[:, 0, :]  # takes [CLS] token representation as input
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class ModelWrapper(pl.LightningModule):
    def __init__(self, args):
        super(ModelWrapper, self).__init__()
        self.config_args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if args.use_pretrained:
            # using pretrained transformers
            self.model = AutoModel.from_pretrained(args.model_name, hidden_dropout_prob=args.dropout_rate, add_pooling_layer=False, cache_dir="/tmp/hugginface")
        else:
            # training transformer from scratch
            self.model = AutoModel.from_config(AutoModel.from_pretrained(
                args.model_name, hidden_dropout_prob=args.dropout_rate, add_pooling_layer=False))
        self.task_head = TexClassificationHead(self.model.config.hidden_size, 3, args.dropout_rate)
        #metrics
        self.train_metric = pl.metrics.Accuracy()
        self.val_metric = pl.metrics.Accuracy()

    def forward(self, input_ids, attention_mask):
        # loading to cuda devices
        # input_seq = input_seq.to(self.transformer.device)
        # attention_mask = attention_mask.to(self.transformer.device)
        # calculating the output logits
        doc_rep = self.model(input_ids, attention_mask=attention_mask)[0]
        output_logits = self.task_head(doc_rep)
        return output_logits
        
    def configure_optimizers(self):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config_args.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config_args.learning_rate)
        # optimizer = Adafactor(optimizer_grouped_parameters, lr=self.config_args.learning_rate, 
        #                                                   scale_parameter=False, relative_step=False, warmup_init=False)
        
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

    def _step(self, batch, step_type):
        if step_type == 'train':
            step_metric = self.train_metric
        else:
            step_metric = self.val_metric
        
        input_ids, attention_mask = batch[:-1]
        model_output = self(input_ids, attention_mask)
        
        if step_type!='test':
            label_ids = batch[-1]

        return_map = {}
        online_logger_data = {}
        pbar = {}
        if step_type!='test':
            task_loss = F.cross_entropy(model_output, label_ids.long())
            acc = step_metric(model_output.softmax(dim=-1), label_ids.long())
            if step_type == 'val':
                return_map['val_loss'] = task_loss
                return_map['val_acc'] = acc 
            else:
                return_map['loss'] = task_loss
                pbar['acc'] = acc

            # updating the online logger
            online_logger_data.update(pbar)
            online_logger_data.update(return_map)
            self.logger.log_metrics(online_logger_data)

            if len(pbar):
                return_map['progress_bar'] = pbar
        return return_map

    def _epoch_end(self, step_outputs, end_type):
        if end_type == 'train':
            end_metric = self.train_metric
        else:
            end_metric = self.val_metric
        
        loss_label = 'loss'
        if end_type == 'val':
            loss_label = 'val_loss'

        if end_type!='test':
            avg_loss = torch.stack([x[loss_label] for x in step_outputs]).mean()
            overall_acc = end_metric.compute()
            self.config_args.logger.info('epoch : %d - average_%s_loss : %f, overall_%s_acc : %f' % (self.current_epoch, end_type, avg_loss.item(),
                                                                                                            end_type, overall_acc.item()))
            # logging to weight and bias if online mode is enabled
            self.logger.log_metrics(
                {'avg_%s_loss' % end_type: avg_loss, 'overall_%s_acc' % end_type: overall_acc})
            self.log('avg_%s_loss' % end_type, avg_loss, prog_bar=True)
            self.log('overall_%s_acc' % end_type, overall_acc, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def training_epoch_end(self, train_step_outputs):
        self._epoch_end(train_step_outputs, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def validation_epoch_end(self, val_step_outputs):
        self._epoch_end(val_step_outputs, 'val')


class TextDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.args = args

    def val_dataloader(self):
        dev_file_path = os.path.join(os.path.abspath(args.dataset_path), 'dev.jsonl')
        val_dataset = get_dataset_loaders(self.tokenizer, dev_file_path, self.args.logger, dataset_count=self.args.val_dataset_count, 
                                          batch_size=self.args.batch_size, max_seq_len=self.args.max_seq_len)
        return val_dataset

    def train_dataloader(self):
        train_file_path = os.path.join(os.path.abspath(args.dataset_path), 'train.jsonl')
        train_dataset = get_dataset_loaders(self.tokenizer, train_file_path, self.args.logger, dataset_count=self.args.train_dataset_count,
                                            batch_size=self.args.batch_size,  max_seq_len=self.args.max_seq_len)
        return train_dataset

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

    final_checkpoint_path = os.path.join(args.checkpoint_path, model_name)
    os.makedirs(final_checkpoint_path, exist_ok=True)

    # Load datasets
    dm = TextDataModule(args)

    call_back_parameters = {
        'filepath': final_checkpoint_path,
        'save_top_k': 1,
        'verbose': True,
        'monitor': 'overall_val_acc',
        'mode': 'max',
    }

    # checkpoint callback to used by the Trainer
    checkpoint_callback = ModelCheckpoint(**call_back_parameters)

    model = ModelWrapper(args)

    args.logger.debug(model)
    args.logger.info('Model has %d trainable parameters' %
                     count_parameters(model))

    callback_list = []

    precision_val = 16 if args.fp16 > 0 else 32

    trainer = pl.Trainer(callbacks=callback_list, max_epochs=args.epochs, min_epochs=1, gradient_clip_val=args.clip_grad_norm,
                         gpus=args.gpus, checkpoint_callback=checkpoint_callback, distributed_backend='ddp', logger=args.online_logger,
                         precision=precision_val, plugins="deepspeed_stage_2")
    # finally train the model
    args.logger.debug('about to start training loop...')
    trainer.fit(model, dm)
    if args.online_mode > 0:
        _, checkpoint_file = get_checkpoint_file(final_checkpoint_path, args.logger)
        args.online_logger.experiment.save(checkpoint_file, base_path=final_checkpoint_path)
    args.logger.debug('training done.')


if __name__ == "__main__":
    parser = ArgumentParser()

    default_checkpoint_path = os.path.join(base_dir, 'lightning_checkpoints')
    
    # Global model configuration
    parser.add_argument('--checkpoint_path', default=default_checkpoint_path, type=str,
                        help='directory where checkpoints are stored')
    parser.add_argument('--dataset_path', required=True, type=str,
                        help='directory where dataset exits')
    parser.add_argument('--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--epochs', default=5, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='adjust batch size per gpu')
    parser.add_argument('--learning_rate', default=2e-5, type=float,
                        help='specify the learning rate')
    parser.add_argument('--clip_grad_norm', default=0.0, type=float,
                        help='clip gradients with norm above specified value, 0 value will disable it.')
    parser.add_argument('--weight_decay', default=0.01, type=float,
                        help='specify the weight decay.')
    parser.add_argument('--dropout_rate', default=0.1, type=float,
                        help='specify the dropout rate for all layer, also applies to all transformer layers.')
    # parser.add_argument('--seed', default=42, type=int,
    # help='seed value for random initialization.')
    parser.add_argument("--enable_scheduler", action='store_true',
                        help='activates the linear decay scheduler.')
    parser.add_argument("--warmup_steps", default=0.01, type=float,
                        help="percentage of total step used as linear warmup while training the model.")
    parser.add_argument('--max_seq_len', type=int, default=200,
                        help="specify the maximum sequence length for processed dataset.")
    # below three arguments are for debugging purpose
    parser.add_argument("--train_dataset_count", type=int, default=0,
                        help="specify number of training data to use. (for debugging purpose). If zero then takes all the available dataset.")
    parser.add_argument("--val_dataset_count", type=int, default=0,
                        help="specify number of validation data to use. (for debugging purpose). If zero then takes all the available dataset.")
    # logger configs
    parser.add_argument('--online_mode', default=0, type=int,
                        help='disables weight and bias syncronization if 0 is passed')
    # if "combined" architecture is active, we can disable mtl (on textual entailment data) associated with it
    parser.add_argument('--model_name', type=str, default='google/muril-base-cased',
                        help='specify pretrained transformer model to use.')
    parser.add_argument('--use_pretrained', type=int, default=1,
                        help='loads pretrained transformer model.')
    # GPU memory utilization optimizations
    parser.add_argument('--fp16', type=int, default=1,
                        help='enable the automatic mixed precision training')
    args = parser.parse_args()

    args.logger_exp_name = "%s-%s-%s" % (args.model_name, args.epochs, args.learning_rate)
    args.logger_exp_name = args.logger_exp_name.replace('/', '-')

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
        'project': 'stage-II-NLI-based-models',    # first create a project on weight & bias with local account
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

    # required for lr_scheduler: count train and val data instances
    start_training(args)
