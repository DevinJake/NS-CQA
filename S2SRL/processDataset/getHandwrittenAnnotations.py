# coding:utf-8
'''Get boolean logs from annotation file 'train_bool_all.txt'.
'''
from itertools import islice
import json

LINE_SIZE = 100000


# To arrange the order of entities in input sequence.
def rearrangeEntities(context_entities, context):
    # TODO: A bug remain to be fixed.
    # If one entity is contained in another entity the alg is not correct.
    # For instance, 'Q910670|Q205576|Q33' in context 'Q333 does Q910670 share the border with Q205576'.
    entity_index = {x: (context.find(x) if context.find(x) != -1 else 100000) for x in context_entities}
    entity_index = sorted(entity_index.items(), key=lambda item: item[1])
    entities_output_string = ','.join([x[0].strip() for x in entity_index])
    # print (entity_index)
    # print (entities_output_string)
    return entities_output_string


# Get logs for boolean.
def getHandwrittenAnnotations(type, read_file_path):
    annotation_lines = list()
    json_file = '../../data/demoqa2/' + str(type) + '.json'
    auto_log_file = '../../data/annotation_logs/' + str(type) + '_auto.log'
    orig_log_file = '../../data/annotation_logs/' + str(type) + '_orig.log'
    fw = open(json_file, 'w', encoding="UTF-8")
    fwAuto = open(auto_log_file, 'w', encoding="UTF-8")
    fwOrig = open(orig_log_file, 'w', encoding="UTF-8")
    with open(read_file_path, 'r', encoding="UTF-8") as infile:
        count = 0
        while True:
            lines_gen = list(islice(infile, LINE_SIZE))
            if not lines_gen:
                break
            for line in lines_gen:
                annotation_lines.append(line.strip())
            count = count + 1
            print(count)
    line_index = 0
    question_dict = {}
    question_dict_dict = {}
    index = 0
    while line_index < len(annotation_lines):
        if annotation_lines[line_index].strip().isdigit():
            index = int(annotation_lines[line_index].strip())
        elif 'context_utterance:' in annotation_lines[line_index]:
            question_dict.setdefault('context_utterance', annotation_lines[line_index].split(':')[1].strip())
        elif 'context_relations:' in annotation_lines[line_index]:
            question_dict.setdefault('context_relations', annotation_lines[line_index].split(':')[1].strip())
        elif 'context_entities:' in annotation_lines[line_index]:
            question_dict.setdefault('context_entities', annotation_lines[line_index].split(':')[1].strip())
        elif 'context_types:' in annotation_lines[line_index]:
            question_dict.setdefault('context_types', annotation_lines[line_index].split(':')[1].strip())
        elif 'context:' in annotation_lines[line_index]:
            question_dict.setdefault('context', annotation_lines[line_index].split(':')[1].strip())
        elif 'orig_response:' in annotation_lines[line_index]:
            question_dict.setdefault('orig_response', annotation_lines[line_index].split(':')[1].strip())
        elif 'response_entities:' in annotation_lines[line_index]:
            question_dict.setdefault('response_entities', annotation_lines[line_index].split(':')[1].strip())
        elif 'CODE:' in annotation_lines[line_index]:
            question_dict.setdefault('CODE', list())
        elif 'symbolic_seq.append' in annotation_lines[line_index]:
            question_dict['CODE'].append(eval(annotation_lines[line_index].split('(')[1].strip().split(')')[0].strip()))
        elif '----------------------------' in annotation_lines[line_index]:
            if (line_index + 1 < len(annotation_lines) and annotation_lines[line_index + 1].strip().isdigit()) or (
                    line_index + 1 == len(annotation_lines)):
                question_dict_dict.setdefault(index, question_dict)
                question_dict = {}
                # print(line_index)
        line_index += 1
    fw.writelines(json.dumps(question_dict_dict, indent=1, ensure_ascii=False))
    print("Writing to %s is done!" % json_file)
    fw.close()
    print("Get information from train_bool_all.txt!")

    question_index = 0
    fwOrig_lines = list()
    fwAuto_lines = list()
    for key, value in question_dict_dict.items():
        if len(value['CODE']) != 0:
            fwOrig_lines.append(str(question_index).strip() + '\n')
            fwOrig_lines.append('context_utterance:' + str(value['context_utterance']).strip() + '\n')
            fwAuto_lines.append(str(question_index).strip() + ' ' + str(value['context_utterance']).strip() + '\n')
            context_entities = [x.strip() for x in value['context_entities'].split('|')]
            context = str(value['context']).strip()
            fwOrig_lines.append('context_entities:' + rearrangeEntities(context_entities, context) + '\n')
            fwOrig_lines.append('context_relations:' + str(value['context_relations']).strip() + '\n')
            fwOrig_lines.append('context_types:' + str(value['context_types']).strip() + '\n')
            fwAuto_lines.append(str(value['CODE']) + '\n')
            question_index += 1
    fwAuto.writelines(fwAuto_lines)
    print("Writing to %s is done!" % auto_log_file)
    fwOrig.writelines(fwOrig_lines)
    print("Writing to %s is done!" % orig_log_file)
    fwAuto.close()
    fwOrig.close()


def getLines(file):
    lines = list()
    with open(file, 'r', encoding="UTF-8") as infile:
        while True:
            lines_gen = list(islice(infile, LINE_SIZE))
            if not lines_gen:
                break
            for line in lines_gen:
                lines.append(line.strip())
    return lines


def combineAnnotations(target_annotate_file, source_annotate_file, orig_file):
    target_annotate_lines = getLines(target_annotate_file)
    source_annotate_lines = getLines(source_annotate_file)
    orig_lines = getLines(orig_file)

    # Analyze auto annotations.
    id_question_action_infos = dict()
    index = 0
    while index < len(target_annotate_lines):
        line_string = str(target_annotate_lines[index]).strip()
        if not line_string.startswith('['):
            string_list = str(line_string).split(' ')
            id = int(string_list[0])
            question = ' '.join(string_list[1:])
            actions = list()
            index += 1
            while index < len(target_annotate_lines) and str(target_annotate_lines[index]).strip().startswith('['):
                actions.append(str(target_annotate_lines[index]).strip())
                index += 1
            question_action_info = dict()
            question_action_info.update({'question': question, 'actions': actions})
            id_question_action_infos[id] = question_action_info

    # Analyze origs.
    id_question = dict()
    index = 0
    while index < len(orig_lines):
        if index % 5 == 0:
            line_string = str(orig_lines[index]).strip()
            id = int(line_string)
            index += 1
            question = str(orig_lines[index]).strip().split(':')[1].strip()
            id_question[id] = question
        else:
            index += 1

    # Analyze manual annotations.
    question_actions = dict()
    index = 0
    while index < len(source_annotate_lines):
        line_string = str(source_annotate_lines[index]).strip()
        if not line_string.startswith('['):
            string_list = str(line_string).split(' ')
            question = ' '.join(string_list[1:])
            actions = list()
            index += 1
            while index < len(source_annotate_lines) and str(source_annotate_lines[index]).strip().startswith('['):
                actions.append(str(source_annotate_lines[index]).strip())
                index += 1
            question_actions[question] = actions

    # Combine auto_annotations and manual_annotations.
    for q, a in question_actions.items():
        flag = True
        for id, info in id_question_action_infos.items():
            if info['question'] == q:
                # Substitute the annotations.
                info['actions'] = a
                flag = False
        # No annotation is found.
        if flag:
            for id, question in id_question.items():
                if q == question:
                    question_action_info = dict()
                    question_action_info.update({'question': q, 'actions': a})
                    id_question_action_infos[id] = question_action_info
                    break

    return id_question_action_infos


def write_to_log(infos, t):
    auto_log_file = '../../data/annotation_logs/' + str(t) + '_auto.log'
    fwauto = open(auto_log_file, 'w', encoding="UTF-8")
    fwauto_lines = list()
    for ID, info in infos.items():
        fwauto_lines.append(str(ID) + ' ' + info['question'].strip() + '\n')
        if len(info['actions']) > 0:
            for action in info['actions']:
                fwauto_lines.append(str(action).strip() + '\n')
    fwauto.writelines(fwauto_lines)
    print("Writing to %s is done!" % auto_log_file)
    fwauto.close()


# Run to get the logs of manually annotation for boolean questions.
if __name__ == "__main__":
    # getHandwrittenAnnotations('boolean', '../../data/demoqa2/train_bool_all.txt')
    getHandwrittenAnnotations('quantitative_count_hand', '../../data/demoqa2/train_count_all.txt')
    id_question_action_infos = combineAnnotations('../../data/annotation_logs/count_auto.log',
                                                  '../../data/annotation_logs/quantitative_count_hand_auto.log',
                                                  '../../data/annotation_logs/count_orig.log')
    write_to_log(id_question_action_infos, 'count_combine')

    getHandwrittenAnnotations('quantitative_hand', '../../data/demoqa2/train_quanti_all.txt')
    id_question_action_infos = combineAnnotations(
        '../../data/annotation_logs/quantative_auto.log', '../../data/annotation_logs/quantitative_hand_auto.log', '../../data/annotation_logs/quantative_orig.log')
    write_to_log(id_question_action_infos, 'quantitative_combine')

