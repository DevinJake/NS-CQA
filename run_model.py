# -*- coding: utf-8 -*-
# @Time    : 2019/1/14 16:00

from Preprocess.prepare_data import PrepareData
from params import get_params
from Preprocess.load_wikidata import load_wikidata

def run_training(param):
    pass

def pre_process(param):
    """
    search entities in the question and then get related triples in wikidata
    :return:
    """
    #wikidata, reverse_dict, prop_data, child_par_dict, wikidata_fanout_dict = load_wikidata(param["wikidata_dir"])
    preparedata = PrepareData(param)
    qadata = preparedata.prepare_data(param["train_dir"],param["train_dir"]+"/out",)
    pass


def main():
    param = get_params("/data/zjy/csqa_data", "/home/zhangjingyao/preprocessed_data_10k")
    pre_process(param)


if __name__=="__main__":
    # with codecs.open('/data/zjy/csqa_data/wikidata_dir/wikidata_short_1.json', 'r', 'utf-8') as data_file:
    #                  '/data/zjy/caqa_data/wikidata_dir/wikidata_short_1.json'
    #     lines = data_file.readlines()
    #     for l in lines:
    #         print(l)
    # pass
    main()