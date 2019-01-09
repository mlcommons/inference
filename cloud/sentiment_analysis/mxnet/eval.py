import argparse
from tqdm import tqdm
import re
import itertools
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
#from data import create_data,pad_sequences
import mxnet as mx
import os
import pickle
import time
from data import SentimentIter
from models.model_cnn import sent_model

DATAPATH='./'
#DATAPATH='/media/drive/sentiment/'
parser = argparse.ArgumentParser(description='Semtiment training')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-embedding', default=1024, type=int, help='Number of workers used in data-loading')
parser.add_argument('--hidden-size', default=1024, type=int, help='Hidden size of RNNs')
parser.add_argument('--eval', default=False, help='Location to save epoch models')
parser.add_argument('--token', default='spacy', help='use spacy tokenizer or not')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--max-seq-len', default=1000, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--vocab-size', default=5200, type=int, help='Annealing applied to learning rate every epoch')
parser.add_argument('--model', default='rnn', help='Model type cnn or rnn')
parser.add_argument('--calc_accuracy', dest='calc_accuracy', action='store_true', help='Calc accuracy on the full validation dataset')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--epoch', type=int, default=0)

def main():
    global args, best_prec1
    args = parser.parse_args()
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    test_iter=SentimentIter(data_path='./Datasets',data_shapes=(args.batch_size,args.max_seq_len), label_shapes=(args.batch_size,2),calc_accuracy=args.calc_accuracy,batch_size=args.batch_size)

    if args.cuda:
        gpu_ids = [int(g) for g in args.gpus.split(',')]
        print('Using GPUs: {}'.format(gpu_ids))
        xpu = mx.gpu(device_id=gpu_ids[0]) if gpu_ids else mx.cpu()
    else:
        xpu=mx.cpu()    

    ##Create Module

    mod = mx.mod.Module(*sent_model(vocab_size=args.vocab_size,emb_dim=args.num_embedding,num_hidden=args.hidden_size,num_classes=2,batch_size=args.batch_size),context=xpu)


    def evaluate_accuracy_fit(label,pred):

        acc = mx.metric.Accuracy()
        predictions = pred.argmax(1)
        acc=1.0-np.abs(predictions-label.argmax(1)).sum()/len(label)
        return acc

    if args.eval:
        print('--------------Running Evaluation -------------')
        sym, arg_params, aux_params = mx.model.load_checkpoint(args.eval, args.epoch)
        mod.bind(data_shapes=test_iter.provide_data, label_shapes=test_iter.provide_label,for_training=False)
        mod.set_params(arg_params=arg_params,aux_params=aux_params,allow_missing= False)

        start_time = time.time()
        acc_test=0
        ii=0   
        if args.calc_accuracy:
            #test_iter_acc =  test_iter.get_acc_iter()
            start_time = time.time()
            num_samples=len(test_iter.all_labels)
            for i, batch in enumerate(test_iter):
                if i%10==0 and i!=0:
                    print('test: %s %%' % (100*i*args.batch_size/num_samples) ,acc_test/i)  
                batch.data[0]=batch.data[0].as_in_context(xpu)
                batch.label[0]=batch.label[0].as_in_context(xpu)
                target=batch.label[0]
                mod.forward(batch, is_train=False)
                pred=mod.get_outputs()[0].asnumpy()
                acc_test+=evaluate_accuracy_fit(target.asnumpy(),pred)
            acc_test/=(i+1)
            end_time = time.time()
            print("Final test_acc %s ,Time %s" %(acc_test,end_time - start_time))  
        else:    
            while True:
                start_time_iter = time.time()
                ii+=1
                batch=test_iter.next()
                if ii%10==0 and ii!=0:
                    print('Tested %s batches with average accuracy: %s' % (ii ,acc_test/ii))
                batch.data[0]=batch.data[0].as_in_context(xpu)
                batch.label[0]=batch.label[0].as_in_context(xpu)
                target=batch.label[0]
                mod.forward(batch, is_train=False)
                pred=mod.get_outputs()[0].asnumpy()
                end_time_iter = time.time()
                print('Time for current iteration: %s ' %(end_time_iter-start_time_iter))
                acc_test+=evaluate_accuracy_fit(batch.label[0].asnumpy(),pred)
    
            acc_test/=(ii+1)
            end_time = time.time()
            print("Final test_acc %s ,Time %s" %(acc_test,end_time - start_time))

if __name__ == '__main__':
    main()
