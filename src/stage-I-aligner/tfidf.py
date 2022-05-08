import stanza
import string
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from numpy import dot
from numpy.linalg import norm
from transformers import AutoTokenizer

class TfIdf:
    def _load_stop_words(self):
        # using dictionary implementation for faster execution
        punct = {x:'' for x in string.punctuation}
        self.stop_words = punct
        if self.lang == 'en':
            from nltk.corpus import stopwords
            self.stop_words.update({x:'' for x in stopwords.words('english')})
        if self.lang == 'hi':
            from spacy.lang.hi import STOP_WORDS as hi_stop_words
            self.stop_words.update({x:'' for x in hi_stop_words})
    
    def _feature_vectorizer(self, doc_corpus, n_gram=1, enable_idf=False):
        # obtain n-grams
        if enable_idf:
            vectorizer = TfidfVectorizer(ngram_range = (n_gram, n_gram), tokenizer=lambda text: self._tokenizer(text))
        else:
            vectorizer = CountVectorizer(ngram_range = (n_gram, n_gram), tokenizer=lambda text: self._tokenizer(text)) 
        doc_vectorizer = vectorizer.fit(doc_corpus)  
        doc_features = (vectorizer.get_feature_names()) 
        return doc_vectorizer, doc_features
    
    def _calculate_score(self, a, b):
        if dot(a, b)==0 or norm(a)==0 or norm(b)==0:
            return 0
        return dot(a, b)/(norm(a)*norm(b))

    def _load_lemmatizer(self):
        if self.lang == 'en':
            self.nlp = stanza.Pipeline(lang=self.lang, processors='tokenize', tokenize_no_ssplit=True)
        elif self.lang == 'hi':
            self.nlp = stanza.Pipeline(lang=self.lang, processors='tokenize', tokenize_no_ssplit=True)
        elif self.lang == 'mr':
            self.nlp = stanza.Pipeline(lang=self.lang, processors='tokenize', tokenize_no_ssplit=True)
        else:
            raise Exception('%s language support is not available yet !!!' % self.lang)
    
    def _handle_ngram_args(self):
        assert self.ngram>0, "invalid ngrams specified, it must be greater than zero ( >0 )"
        if self.ngram_weights is None:
            self.ngram_weights = ((1.0/self.ngram) * np.ones(self.ngram)).tolist()
        else:
            assert isinstance(self.ngram_weights, list), "invalid input for ngram_weights, expected list"
            assert len(self.ngram_weights)==self.ngram, "ngram_weights distribution : %s \
                                    doesn't match with ngram specified : %s" % (self.ngram_weights, self.ngram)
            assert np.sum(self.ngram_weights)==1, "sum of entries of ngram_weight should be one."
    
    def _tokenizer(self, text):
        # as self._process_text has already lemmatized it, we just need to return space separated tokens
        return text.split()
    
    def _process_text(self, text):
        text = self._normalize_input(text)
        if self.use_transformer:
            t_doc = self.tokenizer.tokenize(text)
            return ' '.join(t_doc)
        else:
            t_doc = self.nlp(text)
            return ' '.join([word.text for sent in t_doc.sentences for word in sent.words if word.text not in self.stop_words])
        
    
    def __init__(self, lang, ngram=1, ngram_weights=None, use_idf=True, transformer=False):
        self.lang = lang
        self.ngram = ngram
        self.ngram_weights=ngram_weights
        self.use_idf=use_idf
        self.use_transformer = transformer
        
        self._handle_ngram_args()
        if self.use_transformer:
            self.tokenizer = AutoTokenizer.from_pretrained('google/muril-base-cased')
        else:
            self._load_lemmatizer()
            self._load_stop_words()  
    
    def _normalize_input(self, text):
        text = text.strip()
        text = re.sub(re.compile("\s{2,}"), ' ', text)
        if self.lang == 'en':
            text = text.lower()
        return text
    
    def get_scores(self, query, docs):
        """
        returns tf-idf score for list of documents against the single query string
        
        Args:
            query: a single query instance 
            docs: list of documents on which tf-idf score are calculated
        
        Returns:
            scores: list of tf-idf scores in the same order as of the docs.
        """
        processed_docs = list(map(self._process_text, docs))
        query = self._process_text(query)
        
        final_scores = np.zeros(len(docs)).tolist()
        for ngram in range(1, self.ngram+1):
            vectorizer, _ = self._feature_vectorizer(processed_docs, n_gram=ngram, enable_idf=self.use_idf)
            doc_vector = vectorizer.transform(processed_docs).toarray()
            query_vector = vectorizer.transform([query]).toarray()[0]
            
            assert len(doc_vector) == len(docs), "doc vector and actual doc are of different size."
            #calculating score
            for idx in range(len(doc_vector)):
                final_scores[idx] += self.ngram_weights[ngram-1] * self._calculate_score(query_vector, doc_vector[idx])
        return final_scores