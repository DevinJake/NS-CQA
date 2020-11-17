# !/usr/bin/env python3
# The file is used to predict the action sequences for full-data test dataset.
import torch
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
MAX_TOKENS = 60
HIDDEN_STATE_SIZE = 128
EMBEDDING_DIM = 50


def tokenizer_encode(tokenizer, sample_txt):
    # tokens = tokenizer.tokenize(sample_txt)
    # token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # print(f' Sentence: {sample_txt}')
    # print(f'   Tokens: {tokens}')
    # print(f'Token IDs: {token_ids}')

    encoding = tokenizer.encode_plus(
        sample_txt,
        max_length=MAX_TOKENS,
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',  # Return PyTorch tensors
        truncation=True
    )
    # print(encoding.keys())
    # print(len(encoding['input_ids'][0]))
    # print(encoding['input_ids'][0])
    # print(len(encoding['attention_mask'][0]))
    # print(encoding['attention_mask'])
    # print(tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]))
    return {'sample_txt': sample_txt,
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']}


class CqaBertModel(nn.Module):
    def __init__(self, decoder_hidden_size, pre_trained_model_name, fix_flag=True):
        super(CqaBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(pre_trained_model_name)
        self.drop = nn.Dropout(p=0.3)
        self.bert_out = nn.Linear(self.bert.config.hidden_size, decoder_hidden_size)
        if fix_flag:
            self.freeze_bert_encoder()
        else:
            self.unfreeze_bert_encoder()

    def bert_encode(self, input_ids, attention_mask):
        last_hidden_state, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output_pool = self.drop(pooled_output)
        output_hidden_states = self.drop(last_hidden_state)
        return self.bert_out(output_pool), self.bert_out(output_hidden_states)

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=PRE_TRAINED_MODEL_NAME)

    # # sample_txt = 'When was I last outside? I am stuck at home for 2 weeks.'
    # # print_tokenizer(tokenizer, sample_txt)
    # sample_txt = '<E> ENTITY1 ENTITY2 </E> <R> RELATION1 </R> <T> TYPE1 </T> <I> </I> who are a native of canada and united states of america'
    # encoded_result = tokenizer_encode(tokenizer, sample_txt)
    # print(encoded_result)
    #
    # # print(tokenizer.sep_token, tokenizer.sep_token_id)
    # # print(tokenizer.cls_token, tokenizer.cls_token_id)
    # # print(tokenizer.pad_token, tokenizer.pad_token_id)
    # # print(tokenizer.unk_token, tokenizer.unk_token_id)
    #
    # bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    # last_hidden_state, pooled_output = bert_model(
    #     input_ids=encoded_result['input_ids'],
    #     attention_mask=encoded_result['attention_mask']
    # )
    # print(last_hidden_state.shape)
    # print(pooled_output.shape)

    # m = nn.Dropout(p=0.2)
    # input = torch.randn(2, 16)
    # print(input)
    # output = m(input)
    # print(output)

    model = CqaBertModel(HIDDEN_STATE_SIZE, PRE_TRAINED_MODEL_NAME, True)
    model = model.to(device)

    input_ids_list = []
    attention_mask_list = []

    sample_txt = '<E> ENTITY1 ENTITY2 </E> <R> RELATION1 </R> <T> TYPE1 </T> <I> </I> who are a native of canada and united states of america'
    encoded_result = tokenizer_encode(tokenizer, sample_txt)
    input_ids = encoded_result['input_ids'].to(device)
    input_ids_list.append(input_ids)
    attention_mask = encoded_result['attention_mask'].to(device)
    attention_mask_list.append(attention_mask)
    print(input_ids.shape)  # batch size x seq length
    print(attention_mask.shape)  # batch size x seq length

    sample_txt = 'When was I last outside? I am stuck at home for 2 weeks.'
    encoded_result = tokenizer_encode(tokenizer, sample_txt)
    input_ids_list.append(encoded_result['input_ids'].to(device))
    attention_mask_list.append(encoded_result['attention_mask'].to(device))

    input_ids_s = torch.cat(input_ids_list, dim=0)
    attention_mask_s = torch.cat(attention_mask_list, dim=0)

    output, output_hidden_states = model.bert_encode(input_ids_s, attention_mask_s)
    print(output.shape)
    print(output)
    print(output_hidden_states.shape)
    print(output_hidden_states)
