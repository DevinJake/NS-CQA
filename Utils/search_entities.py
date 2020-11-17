# -*- coding: utf-8 -*-
# @Time    : 2019/1/19 18:09

import json
import sys, os, lucene
from lucene import *
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader, IndexReader
from org.apache.lucene.index import Term
from org.apache.lucene.search import BooleanClause, BooleanQuery, PhraseQuery, TermQuery
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.util import Version

from Preprocess.load_wikidata import load_wikidata
from params import get_params


class LuceneSearch():
    def __init__(self, lucene_index_dir='/data/zjy/csqa_data/lucene_dir/'):
        lucene.initVM(vmargs=['-Djava.awt.headless=true'])
        directory = SimpleFSDirectory(File(lucene_index_dir))
        self.searcher = IndexSearcher(DirectoryReader.open(directory))
        self.num_docs_to_return = 5
        self.ireader = IndexReader.open(directory)

    def search(self, value):
        query = TermQuery(Term("wiki_name", value.lower()))
        # query = BooleanQuery()
        # query.add(new TermQuery(Term("wikidata_name",v)),BooleanClause.Occur.SHOULD)
        # query.add(new TermQuery(Term("wikidata_name",v)),BooleanClause.Occur.SHOULD)
        scoreDocs = self.searcher.search(query, self.num_docs_to_return).scoreDocs
        pids = []
        for scoreDoc in scoreDocs:
            doc = self.searcher.doc(scoreDoc.doc)
            for f in doc.getFields():
                print f.name(), ':', f.stringValue(), ',  '
                if(f.name() == "wiki_id"):
                    pids.append(f.stringValue())
            print ''
        print '-------------------------------------\n'
        return  pids


if __name__ == "__main__":
    ls = LuceneSearch()
    # ls.search("United States")
    # ls.search("India")# Q15975440 Q2060630 Q1936198  Q668 Q274592
    # ls.search("River")# Q20862204 Q20863315 Q7322321 Q11240974 Q2784912
    pids = ls.search("Yangtze")# Q5099535 Q3674754 Q3447500 Q1364589 Q19601344
    params = get_params("/data/zjy/csqa_data", "/home/zhangjingyao/preprocessed_data_10k")
    wikidata, item_data, prop_data, child_par_dict = load_wikidata(params["wikidata_dir"])  # data for entity ,property, type
    for pid in pids:
        if(pid in wikidata.keys()):
            for prop in wikidata[pid]:
                if prop in prop_data:
                    print(pid+":"+item_data[pid],
                          prop+":"+prop_data[prop],
                          [e+":"+item_data[e] for e in wikidata[pid][prop]])

        else:
            print(pid,"not in wikidata")


    print(0)

