import mxnet as mx
from models.crossentropy import CrossEntropyLoss


def sent_model(vocab_size,emb_dim,num_hidden,num_classes,batch_size):

    """
    Describes an 
    """
    data = mx.symbol.Variable('data')                                                                    #[batch_size x max_seq_length]
    label = mx.symbol.Variable('softmax_label')

    embed = mx.symbol.Embedding(data=data,input_dim=vocab_size,output_dim=emb_dim)                       #[batc_size x max_seq_length x embed_dim]
    embed_tm = mx.sym.SwapAxis(embed, dim1=1, dim2=2)                                                    #[max_seq_length x batch_size x embed_dim]

    conv3=mx.symbol.Convolution(embed_tm,num_filter=num_hidden,kernel=(3,))
    conv3=mx.symbol.LeakyReLU(data=conv3,slope=0.25,act_type='elu')
    conv3=mx.symbol.max(data=conv3,axis=2)
    conv4=mx.symbol.Convolution(embed_tm,num_filter=num_hidden,kernel=(4,))
    conv4=mx.symbol.LeakyReLU(data=conv4,slope=0.25,act_type='elu')
    conv4=mx.symbol.max(data=conv4,axis=2)
    c3c4=mx.symbol.Concat(conv3,conv4,dim=1,name='concat_conv34')

    pred = mx.symbol.FullyConnected(data=c3c4, num_hidden=num_classes,name='pred')                      #[batch_size x hidden_size]

    sm = mx.symbol.softmax(data=pred,axis=1,name='softmax')      #[batch_size x output] in sentiment output=2
    ce=mx.symbol.Custom(data=sm, label=label, op_type='CrossEntropyLoss',name='cross_entropy_loss')
    data_names = ['data']
    label_names = ['softmax_label']
    return ce 
