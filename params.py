# -*- coding: utf-8 -*-
# @Time    : 2019/1/14 16:02

def get_params(dir,qadir):
    param = {}
    data_dir = str(dir)
    qadata_dir = str(qadir)
    # filepath
    param['wikidata_dir'] = data_dir + "/wikidata_dir"
    param['transe_dir'] = data_dir + "/transe_dir"
    param['glove_dir'] = data_dir + "/glove_dir"
    param['train_dir'] =  qadata_dir + "/train"
    # settings
    param['use_gold_entities'] = True
    param['use_gold_relations'] = True
    param['use_gold_types'] = True
    # arguments


    return param
