import argparse
from tqdm import tqdm
import logging
import torch
import torch.nn as nn
import itertools
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from data import create_data,pad_sequences
import os
import pickle
import time
from data import IMDBDataset
from models.model_cnn import sent_model
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

DATAPATH='./'
#DATAPATH='/media/drive/sentiment/'
parser = argparse.ArgumentParser(description='Semtiment training')                   
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--workers', default=8, type=int, help='Number of workers used in data-loading')
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--device-ids', default='0',
                    help='device_ids used for training - e.g 0,1,3')
parser.add_argument('--eval', default=False, help='Location to save epoch models')
parser.add_argument('--save', default='sentiment_cnn_final', help='name of the saved model')
parser.add_argument('--model', default='rnn', help='Model type cnn or rnn')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    _, pred = output.float().max(1)
    return pred.eq(target).sum().float()/batch_size

def main():
    global args, best_prec1
    
    args = parser.parse_args()
    
    model=sent_model(embedding_dim=1024,input_dim=1000,num_hidden=1024, num_classes=2)
    loaded_model=torch.load(args.eval)
    model.load_state_dict(loaded_model.state_dict())
    
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(int(args.device_ids[0]))
        cudnn.benchmark = True
        model.to('cuda',dtype=torch.float)
    else:
        torch.manual_seed_all(args.seed)
        args.device_ids = None  

    logging.basicConfig(filename='example_sentiment.log',level=logging.DEBUG)
    #logging.info("created model with configuration: %s", model_config)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    
    val_data = IMDBDataset(data_path='./Datasets',training=False) 
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)    
    
    # evaluate on validation set
    val_loss,val_acc = validate(val_loader, model, criterion)
    print(val_loss,val_acc)
    logging.info('\n Validation Loss {val_loss:.4f}\t'
                 'Validation Acc {val_acc:.4f}\n'
                 .format(val_loss=val_loss,val_acc=val_acc))
    


def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
    if args.device_ids and len(args.device_ids) > 1:
        model = torch.nn.DataParallel(model, args.device_ids)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    end = time.time()
    phase='TRAINING' if training else 'EVALUATING'
    for i, (input,target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.device_ids is not None:
            target = target.to('cuda')
            input = input.to('cuda', dtype=torch.long)
             
        # compute output
        output = model(input)
        loss = criterion(output, target)
        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        prec1= accuracy(output.data, target)
        losses.update(loss.data.item(), input.size(0))
        acc.update(prec1.item(), input.size(0))
        if training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(),5)
            optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print(phase,'Epoch:', epoch, 'Accuracy:', acc.avg,'Iteration:',i)
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {acc.val:.3f} ({acc.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses,acc=acc))

    return losses.avg, acc.avg



def validate(data_loader, model, criterion):
    with torch.no_grad():
        model.eval()
        return forward(data_loader, model, criterion,
                       training=False, optimizer=None)

if __name__ == '__main__':
    main()
