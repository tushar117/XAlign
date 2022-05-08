import json
import os, sys
from argparse import ArgumentParser
import numpy as np
import random
from tqdm import tqdm 
import torch
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from datetime import datetime
from tfidf import TfIdf 
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler

# required to access the python modules present in project directory
currentdir = os.path.dirname(os.path.realpath(__file__))
projectdir = os.path.dirname(currentdir)
sys.path.append(projectdir)

from logger import ManualLogger

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def fact_str(fact, enable_qualifiers=False):
    fact_str = fact[0:2]
    qualifier_str = [' | '.join(x) for x in fact[2]]
    if enable_qualifiers:
        fact_str.extend(qualifier_str)
    return fact_str

def get_tfidf_edge_score(tfidf_scorer, tsentences, tfacts, enable_qualifers=False):
    edge_scores = []
    processed_facts = [' '.join(fact_str(x, enable_qualifiers=enable_qualifers)) for x in tfacts]
    for i in range(len(tsentences)):
        scores = tfidf_scorer.get_scores(tsentences[i], processed_facts)
        for j, k in enumerate(scores):
            edge_scores.append([i, j, k])
    return edge_scores

def pooled_rep(model_output, attention_mask, reduce='cls'):
    if reduce=='cls':
        return model_output[:, 0, :]
    elif reduce == "mean":
        token_embeddings = model_output #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    elif reduce == 'sum':
        token_embeddings = model_output #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        return sum_embeddings
    else:
        raise Exception('reduce function not present !!!')

def encode_in_batches(model, input_ids, attention_mask, batch_size=32):
    with torch.no_grad():
        dataset = TensorDataset(input_ids, attention_mask)
        dataloader = DataLoader(
                dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)
        temp = []
        for batch in dataloader:
            temp_out = model(batch[0].to(
                    device), batch[1].to(device))[0]
            temp.append(temp_out)
    return torch.cat(temp)

def get_edge_scores(model, tokenizer, tsentences, tfacts, reduce='cls'):
    res = []
    with torch.no_grad():
        enc = tokenizer.batch_encode_plus(tsentences, truncation=True, padding='max_length', max_length=512, return_attention_mask=True, return_tensors='pt')
        #taking the [CLS] token
        # s_out = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))[0]
        s_out  = encode_in_batches(model, enc["input_ids"], attention_mask=enc["attention_mask"])
        sentence_encoding = pooled_rep(s_out, enc["attention_mask"].to(device), reduce=reduce)
                
        processed_facts = [' | '.join(fact_str(x, enable_qualifiers=True)) for x in tfacts]
        fenc = tokenizer.batch_encode_plus(processed_facts, truncation=True, padding='max_length', max_length=512, return_attention_mask=True, return_tensors='pt')
        # f_out = model(input_ids=fenc["input_ids"].to(device), attention_mask=fenc["attention_mask"].to(device))[0]
        f_out  = encode_in_batches(model, fenc["input_ids"], attention_mask=fenc["attention_mask"])
        facts_encoding = pooled_rep(f_out, fenc["attention_mask"].to(device), reduce=reduce)
        
        edge_scores = []        
        for i in range(len(tsentences)):
            scores = F.cosine_similarity(facts_encoding, sentence_encoding[i].unsqueeze(0), 1, 1e-6).cpu().tolist()
            for j, k in enumerate(scores):
                edge_scores.append([i, j, k])
        
        return edge_scores

def predict_edges(model, tokenizer, tf_en_scorer, tf_native_scorer, tsentences, translated_sents, tfacts, translated_facts, reduce='cls', threshold=1.0, max_edge_count=None):
    
    src_syntactic_scores = get_tfidf_edge_score(tf_native_scorer, tsentences, translated_facts)
    tgt_syntactic_scores = get_tfidf_edge_score(tf_en_scorer, translated_sents, tfacts, enable_qualifers=True)
        
    fwd_semantic_scores = get_edge_scores(model, tokenizer, tsentences, tfacts, reduce=reduce)
    bwd_semantic_scores = get_edge_scores(model, tokenizer, translated_sents, translated_facts, reduce=reduce)
    
    hybrid_scores = []
    hybrid_scores_map = defaultdict(lambda: defaultdict(lambda: 0))
    for i in range(len(tsentences)*len(tfacts)):
        # tgt_syntactic_scores[i] + src_syntactic_scores[i] + 
        final_score = [src_syntactic_scores[i][2], tgt_syntactic_scores[i][2], fwd_semantic_scores[i][2], bwd_semantic_scores[i][2]]
        hybrid_scores.append([tgt_syntactic_scores[i][0], tgt_syntactic_scores[i][1], np.mean(final_score)])
        hybrid_scores_map[tgt_syntactic_scores[i][0]][tgt_syntactic_scores[i][1]] = np.mean(final_score)
    
    res = []
    max_edge_count =  max_edge_count if max_edge_count else len(tsentences)*len(tfacts)  
    max_edges = min(int(threshold*len(hybrid_scores)), max_edge_count)
    for u in sorted(hybrid_scores, key=lambda x: x[2], reverse=True):
        if max_edges<1:
            break
        res.append(u)
        max_edges-=1
    return res, hybrid_scores_map

from nltk.stem.porter import *
stemmer = PorterStemmer()

def word_stemmer(list_str):
    res = set()
    for x in list_str:
        temp = ' '.join([stemmer.stem(z.strip()) for z in x.split()])
        res.add(temp)
    return res

def custom_fact_filter(x, fact_list):
    useless_predicates = ['entity_name', 'instance of', 'sex or gender', 'given name', 'family name', "described by source", "on focus list of Wikimedia project", "different from", "topic's main category"]
    # fetch country of citizenship first
    nationality = set([x[1].strip() for x in fact_list if x[0].strip()=="country of citizenship"])
    # get all occupation for  
    occupation = set([x[1].strip() for x in fact_list if x[0].strip()=="occupation"])
    occupation = word_stemmer(occupation)
    
    languages_spoken_written_signed = set([x[1].strip() for x in fact_list if x[0].strip()=="languages spoken, written or signed"])
    
    # remove useless predicates
    if x[0] in useless_predicates:
        return False, x
    # remove country for sport if it is same country of citizenship
    if x[0]=="country for sport" and x[1].strip() in nationality:
        return False, x
    if x[0]=="sport" and ' '.join([stemmer.stem(z.strip()) for z in x[1].split()]) in occupation:
        return False, x 
    # replace "writing language" with "languages spoken, written or signed"
    if x[0]=="writing language":
        if x[1] in languages_spoken_written_signed:
            return False, x
        x[0]=="languages spoken, written or signed"
        return True, x
    return True, x

def execute(config):
    article_map = {}
    config.logger.info('loading the articles...')
    with open(config.input_file, encoding='utf-8') as dfile:
        for line in dfile.readlines():
            temp_json = json.loads(line.strip())
            qid = temp_json['qid']
            del temp_json['qid']
            article_map[qid] = temp_json
    
    config.logger.info('loaded %d articles from file : %s' % (len(article_map), config.input_file))
    config.logger.info('removing non-essential facts')
    
    # preprocess the facts for removal of the non-essential facts
    for key, value in article_map.items():
        processed_en_facts = []
        processed_native_facts = []
        assert len(value['facts']) == len(value['translated_facts'])
        for fact_tuple, native_fact_tuple in zip(value['facts'], value['translated_facts']):
            flag, modified_fact_tuple = custom_fact_filter(fact_tuple, value['facts'])
            if not flag:
                continue
            processed_en_facts.append(modified_fact_tuple)
            processed_native_facts.append(native_fact_tuple)
        article_map[key]['facts'] = processed_en_facts
        article_map[key]['translated_facts'] = processed_native_facts

    # loading the module
    tokenizer = AutoTokenizer.from_pretrained(config.plm_module)
    model = AutoModel.from_pretrained(config.plm_module).to(device)

    edge_threshold = config.edge_threshold
    top_k_facts = config.fact_threshold
    config.logger.info('initiating the stage-I aligner with edge threshold : %0.2f and fact threshold : %d' % (edge_threshold, top_k_facts))
    
    model.eval()
    
    config.logger.info('loading the Term Frequency (TF) module for "en" language')
    tf_en_scorer = TfIdf('en', ngram=2, ngram_weights = [0.8, 0.2], use_idf=False)
    config.logger.info('loading the Term Frequency (TF) module for "%s" language' % config.lang)  
    tf_native_scorer = TfIdf(config.lang, ngram=2, ngram_weights = [0.8, 0.2], use_idf=False, transformer=True)

    #writing output to file
    total_sentences = 0
    retained_sentences = 0
    invalid_articles = 0
    start_time = datetime.utcnow()
    config.logger.info('initaiting the sentence and fact filter...')
    with open(config.output_file, 'w', encoding='utf-8') as dfile:
        for i, qid in enumerate(article_map):
            # print("[ %s ] %d - %d - %d" % (qid, i, len(article_map[qid]['sentences']), len(article_map[qid]['facts'])))
            # if facts or sentences is empty then skip
            if len(article_map[qid]['facts'])==0 or len(article_map[qid]['sentences'])==0:
                invalid_articles+=1
                continue
            total_sentences += len(article_map[qid]['sentences'])
            t_qid_data = article_map[qid]

            
            # hybrid alignment
            with torch.no_grad():
                pred_edges, pred_edge_map = predict_edges(model, tokenizer, tf_en_scorer, tf_native_scorer, t_qid_data['sentences'], t_qid_data['translated_sentences'],
                                                                    t_qid_data['facts'], t_qid_data['translated_facts'],
                                                                    threshold=1.00, reduce='mean')
            
            sent_index = defaultdict(lambda: 0.0)
            for rank, edge in enumerate(pred_edges):
                if edge[2]<edge_threshold:
                    break
                sent_index[edge[0]] = edge[2] if sent_index[edge[0]] < edge[2] else sent_index[edge[0]]

            if (i+1)%config.log_interval==0:
                delta = (datetime.utcnow() - start_time).total_seconds()
                config.logger.info('processsd %d / %d articles in %.2f secs' % (i+1, len(article_map), delta))
                config.logger.info('number of empty articles: %d' % invalid_articles)
                config.logger.info('%d [ %0.2f ] sentences retained out of %d total sentences' % (retained_sentences, retained_sentences/total_sentences, total_sentences))
                start_time = datetime.utcnow()

            retained_sentences+=len(sent_index)
            for sindx, sent_score in sent_index.items():
                res = {
                    'qid': qid,
                    'entity_name': t_qid_data['entity_name'],
                    'sentence': t_qid_data['sentences'][sindx],
                    'native_sentence_section': t_qid_data['native_sentence_sections'][sindx],
                    'translated_sentence': t_qid_data['translated_sentences'][sindx],
                    'sent_index': sindx,
                    'facts': [],
                    'sent_score': "%0.2f" % sent_score,
                    'translated_facts': [],
                }
                count = 1
                for k, v in sorted(pred_edge_map[sindx].items(), key=lambda x: x[1], reverse=True):
                    if count>top_k_facts:
                        break
                    res['facts'].append(t_qid_data['facts'][k])
                    res['translated_facts'].append(t_qid_data['translated_facts'][k])
                    count+=1
                json.dump(res, dfile, ensure_ascii=False)
                dfile.write('\n')

        config.logger.info('=='*30)
        config.logger.info('%d [ %0.2f ] sentences retained out of %d total sentences' % (retained_sentences, retained_sentences/total_sentences, total_sentences))
        config.logger.info('number of empty articles: %d' % invalid_articles)

if __name__ == "__main__":
    args = ArgumentParser()

    default_log_path = os.path.join(currentdir, 'logs') 
    
    args.add_argument('--log_dir', type=str, default=default_log_path, 
                        help='specify logging directory')
    args.add_argument('-i', '--input_file', type=str, required=True, 
                        help='specify the input file for stage-I aligner.')
    args.add_argument('-l', '--lang', type=str, required=True, 
                        help='specify language for stage-I aligener.')                    
    args.add_argument('-o', '--output_file', type=str, required=True,
                        help='specify the output file for stage-I aliger.')
    args.add_argument('-m', '--plm_module', type=str, default="google/muril-large-cased",
                        help='specify the pretrained language model for stage-I aliger.')
    args.add_argument('-e', '--edge_threshold', type=float, default=0.60,
                        help='specify threshold between 0 - 1 to filter the edges.')
    args.add_argument('-t', '--fact_threshold', type=int, default=10,
                        help='specify the number of the facts to retain per sentence.')
    args.add_argument('-s', '--log_interval', type=int, default=100,
                        help='specify the logging interval while processing the articles.')
    config = args.parse_args()

    os.makedirs(os.path.abspath(config.log_dir), exist_ok=True)
    log_file = os.path.abspath(os.path.join(os.path.abspath(config.log_dir), "%s_stage_1_aligner.log"%config.lang))
    logger = ManualLogger('main', log_file, use_stdout=True, overwrite=True)

    config.input_file = os.path.abspath(config.input_file)
    config.output_file = os.path.abspath(config.output_file)
    logger.info('logging the cmd line arguments')
    logger.info('--'*30)
    for arg in vars(config):
        logger.info('%s - %s' % (arg, getattr(config, arg)))
    logger.info('=='*30)
    config.logger = logger

    execute(config)