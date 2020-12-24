# coding=utf-8
import json
if __name__ == '__main__':
    old_dict = json.load(open('webQSP_freebase_subgraph.json'))
    # old_dict = dict()
    to_add_list = json.load(open('to_add_list.json'))
    for item in to_add_list:
        triple_list = item.split('**')
        if len(triple_list) == 3:
            s = triple_list[0].replace('ns:', '')
            r = triple_list[1].replace('ns:', '')
            t = triple_list[2]
            triple_item = ({s: {r: [t]}})
            if s in old_dict and r in old_dict[s] and t not in old_dict[s][r]:
                old_dict[s][r].append(t)
            else:
                old_dict.update(triple_item)

    jsondata = json.dumps(old_dict, indent=1)
    fileObject = open('webQSP_freebase_subgraph_new2.json', 'w')
    fileObject.write(jsondata)
    fileObject.close()
