import re
import itertools
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pickle
from mxnet import  nd
import mxnet as mx
import os
import spacy

spacy_en = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])

def read_files(foldername,path=None):

    sentiments = []
    if path is None:
        path=os.curdir
    filenames = os.listdir(path+foldername)
    for file in filenames:
        with open(path+foldername+file,"r", encoding="utf8") as pos_file:
            data=pos_file.read().replace('\n', '')
            sentiments.append(data)
    return sentiments


#some string preprocessing
def clean_str(string):

    #This removes any special characters from the review
    remove_special_chars = re.compile("[^A-Za-z0-9 ]+")

    #This removes any line breaks and replaces them with spaces
    string = string.lower().replace("<br />", " ")

    return re.sub(remove_special_chars, "", string.lower())


def create_count(word_counter,sentiments,token='spacy'):
    for line in sentiments:
        token_line = tokenizer(line) if token=='spacy' else (clean_str(line)).split()
        for word in token_line:
            if word not in word_counter.keys():
                word_counter[word] = 1
            else:
                word_counter[word] += 1
    return word_counter
#This assigns a unique a number for each word (sorted by descending order
#based on the frequency of occurrence)and returns a word_dict

def create_word_index(word_counter):
    idx = 1
    word_dict = {}
    for word in word_counter.most_common():
        word_dict[word[0]] = idx
        idx+=1
    return word_dict


#This helper function creates a encoded sentences by assigning the unique
#id from word_dict to the words in the input text (i.e., movie reviews)
def encoded_sentences(input_file,word_dict,token='spacy'):
    output_string = []
    for line in input_file:
        output_line = []
        token_line = tokenizer(line) if token=='spacy' else (clean_str(line)).split()
        for word in token_line:
            if word in word_dict:
                output_line.append(word_dict[word])
        output_string.append(output_line)
    return output_string

#This helper function decodes encoded sentences
def decode_sentences(input_file,word_dict):
    output_string = []
    for line in input_file:
        output_line = ''
        for idx in line:
            output_line += idx2word[idx] + ' '
        output_string.append(output_line)
    return output_string

#This helper function pads the sequences to maxlen.
#If the sentence is greater than maxlen, it truncates the sentence.
#If the sentence is less than 500, it pads with value 0.
def pad_sequences(sentences,maxlen=500,value=0,pad=True):
    """
    Pads all sentences to the same length. The length is defined by maxlen.
    Returns padded sentences.
    """
    padded_sentences = []
    for sen in sentences:
        new_sentence = []
        if(len(sen) < maxlen and pad):
            num_padding = maxlen - len(sen)
            new_sentence = np.append(sen,[value] * num_padding)
            padded_sentences.append(new_sentence)
        else:
            new_sentence = sen[:maxlen]
            padded_sentences.append(new_sentence)
    return padded_sentences


cleanr = re.compile('<.*?>')
def cleanhtml(raw_html):
  cleantext = re.sub(cleanr, ' ', raw_html)
  return cleantext

def tokenizer(text): # create a tokenizer function
    text = cleanhtml(text)
    return [token.lower_ for token in spacy_en(text) if token.is_punct == False and token.is_bracket == False]

def create_data(path_processed_data,data_path,vocab_size,word_dict=None,train=True,redo_prepro=False,token='spacy'):
    if not os.path.isfile(path_processed_data) or redo_prepro:
        #Ensure that the path below leads to the location of the positive reviews
        
        #DATAPATH='/media/drive/sentiment/Datasets/'
        data_path+='/Clean_IMDB'
        print('creating data from: ', data_path)
        print('Using tokenizer: ', token)
        if train:
            folder_path = "/train/"
        else:
            folder_path = "/test/"
        foldername = folder_path+"pos/"
        postive_sentiment = read_files(foldername,data_path)

        #Ensure that the path below leads to the location of the negative reviews
        foldername = folder_path+"neg/"
        negative_sentiment = read_files(foldername,data_path)

        ###clean string

        #This labels the 'Positive' reviews as 1' and the 'Negative' reviews as 0
        positive_labels = [1 for _ in postive_sentiment]
        negative_labels = [0 for _ in negative_sentiment]


        all_sentiments = postive_sentiment + negative_sentiment

        all_labels = positive_labels + negative_labels
        if word_dict is None:
            #This creates a dictionary of the words and their counts in entire
            #movie review dataset {word:count} and combine all of the reviews into one dataset and create a word
            #dictionary using this entire dataset
            word_counter = Counter()
            word_counter=create_count(word_counter,all_sentiments,token=token)
            word_dict = create_word_index(word_counter)

        #This creates a reverse index from a number to the word
        idx2word = {v: k for k, v in word_dict.items()}
        #Encodes the positive and negative reviews into sequences of number
        positive_encoded = encoded_sentences(postive_sentiment,word_dict,token=token)
        negative_encoded = encoded_sentences(negative_sentiment,word_dict,token=token)

        all_encoded = positive_encoded + negative_encoded
        #Any word outside of the tracked range will be encoded with last position.
        t_data = [np.array([i if i<(vocab_size-1) else (vocab_size-1) for i in s]) for s in all_encoded]
        processed_dir='/'.join(path_processed_data.split('/')[:-1])
        if not os.path.isdir(processed_dir):
            os.makedirs(processed_dir)
        with open(path_processed_data, 'wb') as handle:
            pickle.dump([t_data,all_labels,word_dict], handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(path_processed_data, 'rb') as handle:
            t_data,all_labels,word_dict = pickle.load(handle)
    return t_data,all_labels,word_dict

def preprocessing_data(path_processed_data,data_path,vocab_size,word_dict=None,redo_prepro=False,token='spacy'):
    if path_processed_data is None or not os.path.isfile(path_processed_data) or redo_prepro:
        #Ensure that the path below leads to the location of the positive reviews
        foldername = "/Clean_IMDB/test/pos/"
        postive_sentiment = read_files(foldername,data_path)
        #Ensure that the path below leads to the location of the negative reviews
        foldername = "/Clean_IMDB/test/neg/"
        negative_sentiment = read_files(foldername,data_path)

        #This labels the 'Positive' reviews as 1' and the 'Negative' reviews as 0
        positive_labels = [1 for _ in postive_sentiment]
        negative_labels = [0 for _ in negative_sentiment]
        all_sentiments = postive_sentiment + negative_sentiment

        all_labels = positive_labels + negative_labels
        #Encodes the positive and negative reviews into sequences of number
        all_encoded = encoded_sentences(all_sentiments,word_dict,token=token)
        #Any word outside of the tracked range will be encoded with last position.
        t_data = [np.array([i if i<(vocab_size-1) else (vocab_size-1) for i in s]) for s in all_encoded]

    return t_data,all_labels,word_dict
    
def create_tok_dict(data_path,train_processed_path,vocab_size,max_seq_len,token='spacy'):
    _,_,word_dict=create_data(train_processed_path,data_path,vocab_size,word_dict=None,train=True,token=token)
    return word_dict

def create_test_iterators(data_path,test_processed_path,vocab_size,max_seq_len,batch_size,token='spacy',word_dict=None):
    #data_path=None,train_processed_path=None,test_processed_path=None,vocab_size=5200,max_seq_len=1000
    word_dict =create_tok_dict(data_path,train_processed_path=None,vocab_size=5200,max_seq_len=1000,token='spacy')   
    X_test,y_test_set,word_dict=preprocessing_data(test_processed_path,data_path,vocab_size,word_dict=word_dict,train=False,token=token)
    #prepare dataiter
    X_test, _, y_test_set, _ = train_test_split(X_test, y_test_set, test_size=0, random_state=42)

    #Below we pad the reviews and convert them to MXNet's NDArray format
    test = nd.array(pad_sequences(X_test, maxlen=max_seq_len, value=0))
    y_test = nd.array(y_test_set)
    y_test = mx.nd.one_hot(y_test,2)

    data_test = {'data' : test}
    label_test = {'softmax_label' : y_test}
    test_iter = mx.io.NDArrayIter(data=data_test, label=label_test, batch_size=batch_size, last_batch_handle='discard', label_name='label')
    return test_iter,test,word_dict

class SentimentIter(mx.io.DataIter):
    def __init__(self, data_names='data', data_shapes=(20,1000), data_gen=None,
                 label_names='softmax_label', label_shapes=(20,2), label_gen=None, num_batches=10000,
                 data_path=None,train_processed_path=None,test_processed_path=None,vocab_size=5200,max_seq_len=1000,batch_size=10,token='spacy',word_dict=None,calc_accuracy=False):       
        self._provide_data = [mx.io.DataDesc(name=data_names,shape=data_shapes)]
        self._provide_label = [mx.io.DataDesc(name=label_names,shape=label_shapes)]
        self.num_batches = num_batches
        self.data_gen = data_gen
        self.label_gen = label_gen
        self.cur_batch = 0
        self.data_path=data_path
        self.test_processed_path=test_processed_path if test_processed_path is not None else data_path+'/Processed/imdb_test_data_v5200_spacy_v2.pickle'
        self.train_processed_path=train_processed_path if train_processed_path is not None else data_path+'/Processed/imdb_train_data_v5200_spacy_v2.pickle'
        self.vocab_size=vocab_size
        self.max_seq_len=max_seq_len
        self.batch_size=batch_size
        self.token=token
        self.read_data_to_dram()
        self.num_samples=len(self.all_sentiments)
        if word_dict is not None:
            self.word_dict = word_dict
        else:
            self.word_dict =create_tok_dict(data_path,self.train_processed_path,vocab_size,max_seq_len,token='spacy')    
        if calc_accuracy:
            X_test,y_test_set,word_dict=preprocessing_data(test_processed_path,data_path,vocab_size,word_dict=self.word_dict,token=token)
            X_test, _, y_test_set, _ = train_test_split(X_test, y_test_set, test_size=0, random_state=42)
            test = nd.array(pad_sequences(X_test, maxlen=max_seq_len, value=0))
            y_test = nd.array(y_test_set)
            y_test = mx.nd.one_hot(y_test,2)

            data_test = {'data' : test}
            label_test = {'softmax_label' : y_test}
            self.test_iter_acc = mx.io.NDArrayIter(data=data_test, label=label_test, batch_size=batch_size, last_batch_handle='discard', label_name='label')
    
    def get_acc_iter(self):
        return self.test_iter_acc

    def read_data_to_dram(self):
        path=self.data_path +'/Clean_IMDB'
        postive_sentiment = read_files(foldername="/test/pos/",path=path)
        negative_sentiment = read_files(foldername="/test/neg/",path=path)
        positive_labels = [1 for _ in postive_sentiment]
        negative_labels = [0 for _ in negative_sentiment]
        self.all_sentiments = postive_sentiment + negative_sentiment
        self.all_labels = positive_labels + negative_labels    

    def __iter__(self):
        return self

    def prepro_data(self,idx):
        data=[self.all_sentiments[i] for i in idx]
        all_encoded = encoded_sentences(data,self.word_dict,self.token)
        t_data = [np.array([i if i<(self.vocab_size-1) else (self.vocab_size-1) for i in s]) for s in all_encoded]
        test = pad_sequences(t_data, maxlen=self.max_seq_len, value=0)
        return [nd.array(test)]
    
    def extract_label(self,idx):
        labels=[self.all_labels[i] for i in idx]
        labels = mx.nd.one_hot(nd.array(labels),2)
        return [labels]

    def reset(self):
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self,idx=None):
        if idx is None:
            idx=np.random.randint(0, self.num_samples, self.batch_size)
        if self.cur_batch < self.num_batches:
            self.cur_batch += 1
            data = self.prepro_data(idx)
            label = self.extract_label(idx)
            return mx.io.DataBatch(data, label)
        else:
            print('Finshed running %s batches' % (self.num_batches))
            return StopIteration   