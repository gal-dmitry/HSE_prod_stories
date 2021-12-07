import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from tqdm import tqdm 
from functools import lru_cache

import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from string import punctuation as PUNCTUATION

LEMMATIZER = WordNetLemmatizer()
STOPWORDS = stopwords.words('english') + ["I'm", "n't"]
SEED = 42

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import TfidfModel
import pyLDAvis.gensim_models

import warnings
warnings.filterwarnings('ignore')



"""
Text preprocessing
"""    
def clean_text(text):        
    re_add = ['\s+', '(```[^`]*```)+']
    for sign in re_add:    
        text = re.sub(sign, ' ', text)
    
    punct_add = ['\"', '+', '-', '~', '`']
    for sign in punct_add:
        text = text.replace(sign, '')

    return text.lower()


def pos_tag(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)
        
    
def tokenize(text_lower):
    tokens = nltk.word_tokenize(text_lower)

    filtered_tokens = []
    for token in tokens:
        token = token.strip()
        token = LEMMATIZER.lemmatize(token, pos_tag(token))
        if token != ' ' and token not in STOPWORDS and token not in PUNCTUATION:
            filtered_tokens.append(token)

    return filtered_tokens


def preprocess_text(text):
    if text is None:
        return []
    
    text_lower = clean_text(text)
    filtered_tokens = tokenize(text_lower)
    return filtered_tokens
    
    
"""
PMI matrix
"""
class PMI(object):        
    def __init__(self,
                 threshold=0.2,
                 window_half_size=2,
                 min_df=5,
                 max_df=1200,
                 alpha=0.75):
        
        self.th, self.whs = threshold, window_half_size
        self.M, self.counts = None, None
        self.min_df, self.max_df = min_df, max_df
        self.alpha = alpha
        self.id2token = []
        self.token2id = {}

            
    @lru_cache(maxsize=100_000_000)
    def _idx(self, token):
        if token not in self.token2id:
            curr_idx = len(self.id2token)
            self.token2id[token] = curr_idx
            self.id2token.append(token)
        else:
            curr_idx = self.token2id[token]
        return curr_idx

    
    def _reset(self):
        self.M, self.counts, self.token2id, self.id2token = None, None, {}, []


    def get_pmi_matrix(self, text_reader):
        self._reset()    
        word_counter = Counter()
        corpus_sent_size = 0

        for sentence in text_reader:
            for c in range(len(sentence)):
                word_counter[sentence[c]] += 1
            corpus_sent_size += 1

        for_exclusion = []

        for key, value in word_counter.items():
            if value < self.min_df or value > self.max_df:
                for_exclusion.append(key)

        for_exclusion = set(for_exclusion)

        M_w_c = defaultdict(lambda: 0)
        total_counts = defaultdict(lambda: 0)

        for sentence in text_reader:
            len_sentence = len(sentence)

            for c in range(len_sentence):
                word = sentence[c]
                if word in for_exclusion:
                    continue

                left = max(c - self.whs, 0)
                right = min(c + self.whs + 1, len_sentence)

                for idx in range(left, right):
                    if idx == c:
                        continue
                    context = sentence[idx]
                    if context not in for_exclusion:
                        M_w_c[(word, context)] += 1
                        total_counts[word] += 1

        # check symmetry
        for (word, context) in M_w_c.keys():
            assert M_w_c[(word, context)] == M_w_c[(word, context)], 'check symmetry'

        # alpha weight
        alpha_delimiter = 0
        for cnt in total_counts.values():
            alpha_delimiter += cnt ** self.alpha

        # PPMI
        for (word, context), value in M_w_c.items():
            P_w_c = value
            P_w = total_counts[word]
            P_c_alpha = (total_counts[context] ** self.alpha) / alpha_delimiter 

            pmi = np.log(P_w_c / (P_w * P_c_alpha) + 1e-8)
            ppmi = max(0, pmi)
            M_w_c[(word, context)] = ppmi

        return M_w_c


"""
Text info extraction
"""
def get_dictionary(text):
    dictionary = Dictionary(text)
    dictionary.filter_extremes()
    return dictionary


def get_corpus(text, dictionary):
    return [dictionary.doc2bow(t) for t in text]


def get_dict_corp(text):
    dictionary = get_dictionary(text)
    corpus = get_corpus(text, dictionary)
    return dictionary, corpus

    
"""
Model
"""
class Lda:
    def __init__(self, text, num_topics=10):
        dictionary, corpus = get_dict_corp(text)

        self.tf_idf = TfidfModel(corpus, id2word=dictionary)
        self.corpus = self.tf_idf[corpus]
        self.model = LdaModel(self.corpus, num_topics, id2word=dictionary, random_state=SEED)
        
        self.num_topics = num_topics
        self.dictionary = dictionary

    
    def predict_proba(self, text):
        dictionary, corpus = get_dict_corp(text)
        _corpus = self.tf_idf[corpus]
        proba = list(self.model.get_document_topics(_corpus))
        return proba

    
    def predict_topics(self, text):
        proba = self.predict_proba(text)
        topics = []
        for t in proba:
            ind = np.argmax([pairs[1] for pairs in t])
            topic = t[ind][0]
            topics.append(topic)
        return topics

        
    def topic2words(self, t, k=10):
        return [pair[0] for pair in self.model.show_topic(t, k)]
            
        
    def coherence(self, pmi_matrix, k=10):
        pmi_t_array = []
        for t in range(self.num_topics):
            pmi_t = 0
            topic_top_k_words = self.topic2words(t, k)
            for i in range(k - 1):
                for j in range(i + 1, k):
                    w_i = topic_top_k_words[i]
                    w_j = topic_top_k_words[j]
                    pmi_t += pmi_matrix[(w_i, w_j)]
            pmi_t_array.append(2 * pmi_t / (k * (k + 1)))
        return np.mean(pmi_t_array)

        
    def change_topics_cnt(self, num_topics):
        self.num_topics = num_topics   
        self.model = LdaModel(self.corpus, num_topics, id2word=self.dictionary, random_state=SEED) 
        
        
"""
Visualization
"""    
def coherence(text, topic_cnt, window_size=5, k=10, comment=""):
    model = Lda(text)
    pmi = PMI(window_half_size=window_size//2)
    pmi_matrix = pmi.get_pmi_matrix(text)
    
    coherence_array = []
    for cnt in tqdm(topic_cnt, comment):
        model.change_topics_cnt(cnt)
        new_coherence = model.coherence(pmi_matrix, k)
        coherence_array.append(new_coherence)
        
    return coherence_array


def topic_cnt_grid_search(summary, description, topic_cnt):
    coherence_summary = coherence(summary, topic_cnt, comment="summary")
    coherence_description = coherence(description, topic_cnt, comment="description")
    plt.plot(topic_cnt, coherence_summary, label='summary')
    plt.plot(topic_cnt, coherence_description, label='description')
    plt.grid(True)
    plt.legend()
    
    
def topic_words_table(model, k=10, topics=None):
    df = pd.DataFrame()
    top = topics if topics is not None else range(model.num_topics)
    for t in top:
        topic_top_k_words = model.topic2words(t, k)
        df[f"topic: {t}"] = topic_top_k_words
    return df  

    
def visualize_topics(lda_model, text=None):
    mdl = lda_model.model
    dct = lda_model.dictionary

    if text is not None:
        _, corpus = get_dict_corp(text)
        crp = lda_model.tf_idf[corpus]
    else:
        crp = lda_model.corpus

    return pyLDAvis.gensim_models.prepare(mdl, crp, dct, mds="mmds")        



"""
Dataframe processing
"""
def preprocess_dataframe(df):
    versions = np.hstack(df['Affected versions'])
    versions.sort()
    last = np.unique(versions)[-5:]
    
    mask = df['Affected versions'].apply(lambda x: len(np.intersect1d(x, last)) > 0)
    new_df = df[mask]
    new_df.reset_index(inplace=True, drop=True)
    
    return new_df


def get_unique_versions(new_df):
    unique = set()
    lst_of_lst = new_df["Affected versions"].to_list()
    for lst in lst_of_lst:
        for year in lst:
            unique.add(year)
    return unique


def get_texts(df):
    summary = df.summary.apply(preprocess_text)
    description = df.description.apply(preprocess_text)
    return summary, description

    
def get_mask(df):
    mask_2020_2 = df['Affected versions'].apply(lambda x: '2020.2' in x)
    mask_2020_3 = df['Affected versions'].apply(lambda x: '2020.3' in x)    
    return mask_2020_2, mask_2020_3


def get_unique_summary(summary, df):
    mask_2020_2, mask_2020_3 = get_mask(df)
    summary_text_2020_2 = summary[mask_2020_2]
    summary_text_2020_3 = summary[mask_2020_3]
    return summary_text_2020_2, summary_text_2020_3
    
    
def set_diff(prediction_2020_2, prediction_2020_3):
    unique_2020_2 = set(prediction_2020_2)
    unique_2020_3 = set(prediction_2020_3)

    only_2020_2 = unique_2020_2 - unique_2020_3
    only_2020_3 = unique_2020_3 - unique_2020_2
    
    return only_2020_2, only_2020_3

    
    
    
    
    