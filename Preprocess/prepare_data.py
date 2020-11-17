# -*- coding: utf-8 -*-
# @Time    : 2019/1/18 15:33

import _pickle as pkl
import gensim


class PrepareData():
    def __init__(self,param):
        transe_dir = param['transe_dir']
        glove_dir = param['glove_dir']
        # id_entity_map = {0: '<pad_kb>', 1: '<nkb>'}
        # id_entity_map.update(
        #     {(k + 2): v for k, v in pkl.load(open(transe_dir + '/id_ent_map.pickle', 'rb')).iteritems()})

        glove_model = gensim.models.KeyedVectors.load_word2vec_format(glove_dir + '/GoogleNews-vectors-negative300.bin',
                                                                      binary=True)  # /dccstor/cssblr/amrita/resources/glove/GoogleNews-vectors-negative300.bin', binary=True)
        vocab = glove_model.wv.vocab.keys()
        self.glove_embedding = {v: glove_model.wv[v] for v in vocab}

        pass

