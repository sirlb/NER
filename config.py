# -*- coding: utf-8 -*-


base_path = '/home/zhongsd/lb/pretraining_models/tensorflow/'
#base_path =  '/home/liubo/zm/models/'

def model_backbone_path(model_backbone):
    
    if model_backbone == 'roberta_wwm_ext':
        config_path = base_path + 'chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
        checkpoint_path = base_path + 'chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
        dict_path = base_path + 'chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
    
    elif model_backbone == 'roberta_wwm_ext_kg':
        config_path = base_path + 'chinese_roberta_wwm_ext_L-12_H-768_A-12-kg/bert_config.json'
        checkpoint_path = base_path + 'chinese_roberta_wwm_ext_L-12_H-768_A-12-kg/bert_model.ckpt'
        dict_path = base_path + 'chinese_roberta_wwm_ext_L-12_H-768_A-12-kg/vocab.txt'
        
    elif model_backbone == 'bert':
        config_path = base_path + 'chinese_L-12_H-768_A-12/bert_config.json'
        checkpoint_path = base_path + 'chinese_L-12_H-768_A-12/bert_model.ckpt'
        dict_path = base_path + 'chinese_L-12_H-768_A-12/vocab.txt'
    
    elif model_backbone == 'bert_wwm_ext':
        config_path = base_path + 'chinese_wwm_ext_L-12_H-768_A-12/bert_config.json'
        checkpoint_path = base_path + 'chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
        dict_path = base_path + 'chinese_wwm_ext_L-12_H-768_A-12/vocab.txt'
        
    elif model_backbone == 'bert_wwm':
        config_path = base_path + 'chinese_wwm_L-12_H-768_A-12/bert_config.json'
        checkpoint_path = base_path + 'chinese_wwm_L-12_H-768_A-12/bert_model.ckpt'
        dict_path = base_path + 'chinese_wwm_L-12_H-768_A-12/vocab.txt'    
    
    elif model_backbone == 'roberta_wwm_large_ext':
        config_path = base_path + 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_config.json'
        checkpoint_path = base_path + 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_model.ckpt'
        dict_path = base_path + 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/vocab.txt'
    
    
    return config_path,checkpoint_path,dict_path


def dataset_path(dataset):
    
    if dataset == 'weibo':
        train_path = 'data/ner_dataset/weibo/train.bio'
        dev_path = 'data/ner_dataset/weibo/dev.bio'
        test_path = 'data/ner_dataset/weibo/test.bio'
    
    elif dataset == 'resume':
        train_path = 'data/ner_dataset/resume/train.bio'
        dev_path = 'data/ner_dataset/resume/dev.bio'
        test_path = 'data/ner_dataset/resume/test.bio'
    
    elif dataset == 'msra':
        train_path = 'data/ner_dataset/msra/train.bio'
        dev_path = 'data/ner_dataset/msra/dev.bio'
        test_path = 'data/ner_dataset/msra/test.bio'
    
        
    elif dataset == 'ontonote4':
        train_path = 'data/ner_dataset/ontonote4/train.bio'
        dev_path = 'data/ner_dataset/ontonote4/dev.bio'
        test_path = 'data/ner_dataset/ontonote4/test.bio'
        
 
        
    return train_path,dev_path,test_path