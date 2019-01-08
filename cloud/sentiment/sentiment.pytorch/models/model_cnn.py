import torch
import torch.nn as nn
import torchvision.transforms as transforms

__all__ = ['sent_model']


class Sentiment_CNN(nn.Module):

    def __init__(self, vocab_size,input_dim, embedding_dim,num_hidden,num_classes=2):
        super(Sentiment_CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.clm0 = self.conv_leaky_max(embedding_dim,num_hidden,3)
        self.clm1 = self.conv_leaky_max(embedding_dim,num_hidden,4)
        self.classifier = nn.Linear(2*num_hidden,2)
    def conv_leaky_max(self,embedding_dim,num_hidden,kernel_size,slope=0.25):
        conv_leaky_unit= nn.Sequential(
                        nn.Conv1d(embedding_dim,num_hidden,kernel_size=kernel_size),
                        nn.ELU(alpha=slope,inplace=True),
                        nn.AdaptiveMaxPool1d(1)
                        )
        return conv_leaky_unit              

    def forward(self, x):
        x = self.embedding(x).transpose(2,1)
        x1=self.clm0(x)
        x2=self.clm1(x) 
        xc=torch.cat((x1.squeeze(2),x2.squeeze(2)),1)
        y = self.classifier(xc)
        return y


def sent_model(vocab_size=5200,embedding_dim=1024,input_dim=1000,num_hidden=1024, num_classes=2):
    return Sentiment_CNN(vocab_size,input_dim, embedding_dim,num_hidden,num_classes)
