# -*- coding: utf-8 -*-
import os
os.environ['TF_KERAS'] = '1'
os.environ['RECOMPUTE'] = '1'

import numpy as np
import random
from tqdm import tqdm

from transformer.backend import keras, K
from transformer.models import build_transformer_model
from transformer.tokenizers import Tokenizer
from transformer.optimizers import Adam
from transformer.snippets import sequence_padding, DataGenerator
from transformer.snippets import open, ViterbiDecoder, to_array
from transformer.layers import ConditionalRandomField, GlobalPointer
from transformer.backend import multilabel_categorical_crossentropy

import tensorflow as tf
from tensorflow.keras.layers import (
                                        LSTM, Bidirectional,Conv1D,Dense,Lambda,
                                        BatchNormalization,Dropout,concatenate,
                                        Layer,Multiply,LayerNormalization,SeparableConv1D,
                                        Concatenate
                                    )
from tensorflow.keras.models import Model
from layer import  ResidualGatedConv1D

from config import model_backbone_path,dataset_path

import argparse


def set_global_determinism(seed=0, fast_n_close=False):
    
    set_seeds(seed=seed)
    if fast_n_close:
        return

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    from tfdeterminism import patch
    patch()

def set_seeds(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)



def load_data(filename):
    D = []
    categories = list()
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d = ['']
            for i, c in enumerate(l.split('\n')):
                char, flag = c.split(' ')
                d[0] += char
                if flag[0] == 'B':
                    d.append([i, i, flag[2:]])
                    categories.append(flag[2:])
                elif flag[0] == 'I':
                    d[-1][1] = i
            D.append(d)
    return D, categories



class data_generator(DataGenerator):

    def __iter__(self, random=False):
        batch_size = self.batch_size
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
            mapping = tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            if 'crf' in model_type:
                labels = np.zeros(len(token_ids))
            elif 'globalpointer' in model_type:
                labels = np.zeros((len(categories), maxlen, maxlen))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    if 'crf' in model_type:
                        labels[start] = categories.index(label) * 2 + 1
                        labels[start + 1:end + 1] = categories.index(label) * 2 + 2
                    elif 'globalpointer' in model_type:
                        label = categories.index(label)
                        labels[label, start, end] = 1
            
            if contrastive:
                for i in range(2):
                    batch_token_ids.append(token_ids)
                    batch_segment_ids.append(segment_ids)
                    if 'crf' in model_type:
                        batch_labels.append(labels)
                    elif 'globalpointer' in model_type:
                        batch_labels.append(labels[:, :len(token_ids), :len(token_ids)])
                batch_size = self.batch_size * 2
            else:
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                if 'crf' in model_type:
                        batch_labels.append(labels)
                elif 'globalpointer' in model_type:
                    batch_labels.append(labels[:, :len(token_ids), :len(token_ids)])
            if len(batch_token_ids) == batch_size or is_end:
                seq_dims = len(batch_labels[0].shape)
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels, seq_dims=seq_dims)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []




def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)

def global_pointer_crossentropy(y_true, y_pred):
    bh = K.prod(K.shape(y_pred)[:2])
    y_true = K.reshape(y_true, (bh, -1))
    y_pred = K.reshape(y_pred, (bh, -1))
    loss = K.mean(multilabel_categorical_crossentropy(y_true, y_pred))
    return K.mean(loss) 


def global_pointer_crossentropy_rdrop(y_true, y_pred):
    loss1 = kullback_leibler_divergence(y_pred[::2], y_pred[1::2]) + \
        kullback_leibler_divergence(y_pred[1::2], y_pred[::2])
        
    bh = K.prod(K.shape(y_pred)[:2])
    y_true = K.reshape(y_true, (bh, -1))
    y_pred = K.reshape(y_pred, (bh, -1))
    
    loss2 = K.mean(multilabel_categorical_crossentropy(y_true, y_pred))
    return K.mean(loss1) + K.mean(loss2) * 8


def sparse_loss(y_true, y_pred):
    
    return CRF.sparse_loss(y_true, y_pred)

def sparse_rdrop_loss(y_true, y_pred):
    loss1 = kullback_leibler_divergence(y_pred[::2], y_pred[1::2]) + \
        kullback_leibler_divergence(y_pred[1::2], y_pred[::2])
    
    loss2 = sparse_loss(y_true, y_pred)
    
    return K.mean(loss1) + K.mean(loss2) * 8


def global_pointer_f1_score(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, 0), K.floatx())
    return 2 * K.sum(y_true * y_pred) / K.sum(y_true + y_pred)

def get_loss():
    if 'crf' in model_type:
        if contrastive:
            return sparse_rdrop_loss
        else:
            return sparse_loss
    elif 'globalpointer' in model_type:
        if contrastive:
            return global_pointer_crossentropy_rdrop
        else:
            return global_pointer_crossentropy
        



def get_metrics():
    if CRF is not None:
        return CRF.sparse_accuracy
    else:
        return global_pointer_f1_score        




class Base_Model():
    
    def __init__(self,
                 config_path=None,
                 checkpoint_path=None,
                 **kwargs
                 ):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path        
      
    def build_model(self):
        raise NotImplementedError
    

class BERT_IDGRCNN_BiLSTM_CRF_Model(Base_Model):
    def __init__(self,**kwargs):
        super(BERT_IDGRCNN_BiLSTM_CRF_Model, self).__init__(**kwargs)
        self.blocks = 4
    
    def build_model(self):
        dropout_rate = 0.3 if contrastive else 0
        pretrained_model = build_transformer_model(
            self.config_path,
            self.checkpoint_path,
            dropout_rate=dropout_rate,
            return_keras_model=False
        )
    
        output = pretrained_model.output
        stack_layers = []
        
        for layer_idx in range(self.blocks):
            output = ResidualGatedConv1D(kernel_size=3, 
                                         dilation_rate=1,
                                         skip_connect=True,
                                         drop_gate=0.1,
                                         )(output)
            output = ResidualGatedConv1D(kernel_size=3, 
                                         dilation_rate=2,
                                         skip_connect=True,
                                         drop_gate=0.1,
                                         )(output)
            output = ResidualGatedConv1D(kernel_size=3, 
                                         dilation_rate=4,
                                         skip_connect=True,
                                         drop_gate=0.1,
                                         )(output)
            output = ResidualGatedConv1D(kernel_size=3, 
                                         dilation_rate=8,
                                         skip_connect=True,
                                         drop_gate=0.1,
                                         )(output)
            stack_layers.append(output)
            
        output = concatenate(stack_layers, axis=-1)
        output = Bidirectional(LSTM(128, return_sequences=True))(output)
        output = Dense(len(categories) * 2 + 1)(output)
        CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
        output = CRF(output)
        
        model = Model(pretrained_model.input, output)
        return model,CRF
    

class BERT_CRF_Model(Base_Model):
    def __init__(self,**kwargs):
        super(BERT_CRF_Model, self).__init__(**kwargs)
    
    def build_model(self):
        dropout_rate = 0.3 if contrastive else 0
        pretrained_model = build_transformer_model(
            self.config_path,
            self.checkpoint_path,
            dropout_rate=dropout_rate,
            return_keras_model=False
        )
    
        output = pretrained_model.output
        output = Dense(len(categories) * 2 + 1)(output)
        CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
        output = CRF(output)
        
        model = Model(pretrained_model.input, output)
        return model,CRF
    
   
class GlobalPointer_Model(Base_Model):
    def __init__(self,**kwargs):
        super(GlobalPointer_Model, self).__init__(**kwargs)
    
    def build_model(self):
        dropout_rate = 0.3 if contrastive else 0
        pretrained_model = build_transformer_model(
            self.config_path,
            self.checkpoint_path,
            dropout_rate=dropout_rate,
            return_keras_model=False
        )
    
        output = pretrained_model.output
        output = GlobalPointer(len(categories), 64)(output)

        model = Model(pretrained_model.input, output)
        return model,None
    

def build_ner_model(
                    config_path=None,
                    checkpoint_path=None,
                    model_type='bert_crf',
                    **kwargs
                    ):
   
    models = {
        'bert_crf': BERT_CRF_Model,  
        'bert_idgrcnn_bilstm_crf': BERT_IDGRCNN_BiLSTM_CRF_Model,  
        'globalpointer': GlobalPointer_Model,
    }

    
    MODEL = models[model_type]
    ner_model = MODEL(
                         config_path=config_path,
                         checkpoint_path=checkpoint_path
                      )
    model,crf = ner_model.build_model()

    return model,crf





class NamedEntityRecognizer(ViterbiDecoder):

    def __init__(self,**kwargs):
        if CRF is not None:
            super(NamedEntityRecognizer, self).__init__(trans=K.eval(CRF.trans), starts=[0], ends=[0],**kwargs)
        else:
            super(NamedEntityRecognizer, self).__init__(trans=np.array([]), starts=[0], ends=[0],**kwargs)
    

    def crf_recognize(self, text):
        tokens = tokenizer.tokenize(text, maxlen=512)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = model.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], categories[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False
        return [(mapping[w[0]][0], mapping[w[-1]][-1], l) for w, l in entities]
    
    def globalpointer_recognize(self, text, threshold=0):
        tokens = tokenizer.tokenize(text, maxlen=512)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        scores = model.predict([token_ids, segment_ids])[0]
        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf
        entities = []
        for l, start, end in zip(*np.where(scores > threshold)):
            entities.append(
                (mapping[start][0], mapping[end][-1], categories[l])
            )
        return entities

    def recognize(self, text, threshold=0):
        if 'crf' in model_type: 
            entities = self.crf_recognize(text)
        elif 'globalpointer' in model_type:
            entities = self.globalpointer_recognize(text)
        return entities




def evaluate(data):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=100):
        R = set(NER.recognize(d[0]))
        T = set([tuple(i) for i in d[1:]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    metrics = {
            'precision': round(precision,6),
            'recall': round(recall,6),
            'f1': round(f1,6),
            
            }
    return metrics


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_f1 = 0
        self.best_test_f1 = 0
        
    def on_epoch_end(self, epoch, logs=None):
        if CRF is not None:
            trans = K.eval(CRF.trans)
            NER.trans = trans
        metrics = evaluate(valid_data)
        f1, precision, recall = metrics['f1'],metrics['precision'],metrics['recall']
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights(model_path)
            
            print(
                'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best_val_f1: %.5f\n' %
                (f1, precision, recall, self.best_val_f1)
            )
            
            
            metrics  = evaluate(test_data)
            f1, precision, recall = metrics['f1'],metrics['precision'],metrics['recall']
            
            self.best_test_f1 = f1
            print(
                'test:  f1: %.5f, precision: %.5f, recall: %.5f, best_test_f1: %.5f\n' %
                (f1, precision, recall, self.best_test_f1)
            )
            
        else:
            print(
                'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best_val_f1: %.5f\n' %
                (f1, precision, recall, self.best_val_f1)
            )
            
            metrics  = evaluate(test_data)
            f1, precision, recall = metrics['f1'],metrics['precision'],metrics['recall']
            
            print(
                'test:  f1: %.5f, precision: %.5f, recall: %.5f, best_test_f1: %.5f\n' %
                (f1, precision, recall, self.best_test_f1)
            )
            
def str2bool(str):
    return True if str.lower() == 'true' else False


maxlen = 256
batch_size = 8
crf_lr_multiplier = 1000


parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning_rate', type=float,default=1e-5)
parser.add_argument('-mt', '--model_type', default='bert_idgrcnn_bilstm_crf')
parser.add_argument('-ds', '--dataset', default='math_ate')
parser.add_argument('-rs', '--rand_seed', type=int,default=2022)
parser.add_argument('-mb', '--model_backbone', default='roberta_wwm_ext')
parser.add_argument('-ci', '--cuda_id', default='4')
parser.add_argument('-epoch', '--epochs', type=int,default=20)
parser.add_argument('-contrastive', '--contrastive', type=str2bool,default=True)


args = parser.parse_args()

learning_rate = args.learning_rate
model_type = args.model_type
dataset = args.dataset
rand_seed = args.rand_seed
model_backbone = args.model_backbone
cuda_id = args.cuda_id
epochs = args.epochs
contrastive = args.contrastive



# PATH
config_path, checkpoint_path, dict_path = model_backbone_path(model_backbone)
train_path,dev_path,test_path = dataset_path(dataset)

set_global_determinism(seed=rand_seed)

# gpu
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))



if __name__ == '__main__':

    train_data, train_categorie = load_data(train_path)
    valid_data, valid_categorie = load_data(dev_path)
    test_data, test_categorie = load_data(test_path)
    categories = list(sorted(set(train_categorie+valid_categorie+test_categorie)))
    
    tokenizer = Tokenizer(dict_path, do_lower_case=True)


    
    model_path = f'model_save/{model_type}/best_model.weights'
    
    model,CRF = build_ner_model(config_path,checkpoint_path,model_type=model_type)
    loss = get_loss()
    metrics = get_metrics()
    model.summary()
    model.compile(
        loss=loss,
        optimizer=Adam(learning_rate),
        metrics=[metrics]
    )
    NER = NamedEntityRecognizer()


    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )
    
    
     