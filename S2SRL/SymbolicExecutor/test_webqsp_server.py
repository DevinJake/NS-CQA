from symbolics_webqsp import Symbolics_WebQSP
import json

to_test_by_hand_list = []
with open("to_test_by_hand.json", "r", encoding='UTF-8') as to_test_by_hand_file:
    to_test_by_hand_list = json.load(to_test_by_hand_file)

result_list_fortest = []
with open("result_list_fortest.json", "r", encoding='UTF-8') as to_test_by_hand_file:
    result_list_fortest = json.load(to_test_by_hand_file)

for qa in result_list_fortest:
    for key, value in qa.items():



test_action_seq = []
symbolic_exe = Symbolics_WebQSP(test_action_seq)
answer = symbolic_exe.executor()
print("answer 74: ", answer)
