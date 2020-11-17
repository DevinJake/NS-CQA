# -*- coding: utf-8 -*-
# @Time    : 2019/1/18 15:17

import json
import unicodedata


class QuestionParser():

    def __init__(self, params, all_possible_ngrams):
        # item_wikidata = json.load(open(params["wikidata_dir"] + '/items_wikidata_n.json'))
        # self.item_wikidata = {k: self.clean_string(v) for k, v in item_wikidata.items()}
        self.use_gold_entities = params['use_gold_types']
        self.use_gold_relations = params['use_gold_types']
        self.use_gold_types = params['use_gold_types']
        self.all_possible_ngrams = all_possible_ngrams

    def getNER(self, q):
        if (self.use_gold_entities):
            print("use_gold_entity")
            self.entities = []
            context_enties = q["context_entities"].replace("\n", "")
            if (context_enties != ""):
                self.entities.extend(context_enties.split("|"))
            else:
                self.entities = []
                print("no_gold_entity_exit")
        else:
            print("not_use_gold_entity")
        return self.entities

    def getTypes(self, q):
        if (self.use_gold_types):
            print("use_gold_types")
            self.types = []
            context_types = q["context_types"].replace("\n", "")
            if (context_types != ""):
                self.types.extend(context_types.split("|"))
            else:
                self.types = []
                print("no_gold_type_exit")
        else:
            print("not_use_gold_type")
        return self.types

    def getRelations(self, q):
        if (self.use_gold_relations):
            print("use_gold_relations")
            self.relations = []
            context_relations = q["context_relations"].replace("\n", "")
            if (context_relations != ""):
                self.relations.extend(context_relations.split("|"))
            else:
                self.relations = []
                print("no_gold_type_exit")
        else:
            print("not_use_gold_relation")
        return self.relations

    def clean_string(self, s):
        if isinstance(s, str):
            s = unicodedata.normalize('NFKD', unicode(s)).encode('ascii', 'ignore')
        elif isinstance(s, unicode):
            s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore')
        else:
            raise Exception('unicode error')
        return s
    # subFunction


