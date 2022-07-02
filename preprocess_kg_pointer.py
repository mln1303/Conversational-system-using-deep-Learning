from pytorch_pretrained_bert import BertTokenizer
import torch
import os
import codecs
import json
import glob
import random
import argparse
import pickle
import re
from nltk.corpus import stopwords

nltk_stopwords = stopwords.words('english')


def match(sent1, sent2):
    sent1 = sent1[8:].split()
    sent2 = sent2.split()
    # print('ss1',sent1)
    # print('ss2',sent2)

    common = set(sent1).intersection(set(sent2))
    # print('c',common)
    # print(len(common)/(len(set(sent1))))
    #
    if len(common) / len(set(sent1)) > 0.90:
        # print('True')
        return True
    else:
        return False

def get_entities(kg_enc_input):
    enc_entities = [[triple[2] for triple in utterance] for utterance in kg_enc_input]
    # dec_entities = [[triple[0] for triple in utterance] for utterance in kg_dec_input]

    # print((enc_entities))
    # print((dec_entities))

    # entities = enc_entities + dec_entities

    return enc_entities

def transform_triple_to_hrt(triple):
    """ Transforms triple-idx (as a whole) to h/r/t format """
    h, r, t = triple
    try:
        # print(tokenizer.convert_tokens_to_ids([h]))
    # print('hh',[str(h)])
    # print(t)
        return [vocab_dic[h], relation_dic[r], vocab_dic[t]]
    except:
        vocab_dic[t] = len(vocab_dic)
        return [vocab_dic[h], relation_dic[r], vocab_dic[t]]
    # except:
        # print(h)

        # return [tokenizer.convert_tokens_to_ids(['_NAF_H'][0]), rel2id[r], tokenizer.convert_tokens_to_ids(['_NAF_T'][0])]

def convert_tokens_to_id(kg):
    kg = [transform_triple_to_hrt(triple) for triple in kg]
    return kg


def seq2token_ids(source_seqs, target_seq):
    # 可以尝试对source_seq进行切分
    triples_len = 20
    encoder_input = []
    for source_seq in source_seqs:
        # 去掉 xx：
        # print('sss',source_seq[8:])
        encoder_input += tok.tokenize(source_seq[8:]) + ["[SEP]"]

    decoder_input = ["[CLS]"] + tok.tokenize(target_seq[7:])  # 去掉 xx：
    # print(encoder_input)
    # print(decoder_input)

    # 设置不得超过 MAX_ENCODER_SIZE 大小
    if len(encoder_input) > MAX_ENCODER_SIZE - 1:
        if "[SEP]" in encoder_input[-MAX_ENCODER_SIZE:-1]:
            idx = encoder_input[:-1].index("[SEP]", -(MAX_ENCODER_SIZE - 1))
            encoder_input = encoder_input[idx + 1:]

    encoder_input = ["[CLS]"] + encoder_input[-(MAX_ENCODER_SIZE - 1):]
    decoder_input = decoder_input[:MAX_DECODER_SIZE - 1] + ["[SEP]"]
    enc_len = len(encoder_input)
    dec_len = len(decoder_input)

    # print(encoder_input)
    # print(decoder_input)
    
    kg_enc = return_kg(encoder_input, decoder_input)
    # kg_dec = return_kg(decoder_input)
    # print('kg',len(kg_enc))

    kg_enc_input = convert_tokens_to_id(kg_enc[:400])
    # print(kg_enc_input)

    # entity = [vocab_dic[item] for item in entity]


    # kg_dec_input = convert_tokens_to_id(kg_dec)


    # print('ee',encoder_input)

    # conver to ids
    encoder_input = [vocab_dic[item] for item in encoder_input]
    decoder_input = [vocab_dic[item] for item in decoder_input]
    # print('ee',encoder_input)

    # mask
    mask_encoder_input = [1] * len(encoder_input)
    mask_decoder_input = [1] * len(decoder_input)

    # padding
    encoder_input += [0] * (MAX_ENCODER_SIZE - len(encoder_input))
    decoder_input += [0] * (MAX_DECODER_SIZE - len(decoder_input))
    # entity += [0] * (MAX_ENCODER_SIZE*triples_len - len(entity))

    kg_enc_input += [[0, 0, 0]] * (MAX_ENCODER_SIZE - len(kg_enc_input))
    # kg_dec_input += [[[0, 0, 0]] * triples_len] * (MAX_DECODER_SIZE - len(kg_dec_input))
    # print(kg_enc_input)


    # entity = get_entities(kg_enc_input)

    # entity = torch.LongTensor((entity))
    # print(entity)

    mask_encoder_input += [0] * (MAX_ENCODER_SIZE - len(mask_encoder_input))
    mask_decoder_input += [0] * (MAX_DECODER_SIZE - len(mask_decoder_input))

    # turn into tensor
    encoder_input = torch.LongTensor(encoder_input)
    decoder_input = torch.LongTensor(decoder_input)
    # print(encoder_input)
    # print(kg_input)
    kg_enc_input = torch.LongTensor(kg_enc_input)
    # kg_dec_input = torch.LongTensor(kg_dec_input)



    mask_encoder_input = torch.LongTensor(mask_encoder_input)
    mask_decoder_input = torch.LongTensor(mask_decoder_input)

    return encoder_input, decoder_input, mask_encoder_input, mask_decoder_input, kg_enc_input


def return_kg(enc_words, dec_words):
    NAF = ["_NAF_H", '_NAF_R', "_NAF_T"]
    x = []
    triples_len = 20
    enc_words = set([d_word for d_word in convert_to_original_length(enc_words) if d_word not in nltk_stopwords and len(d_word) > 1])
    dec_words = set([d_word for d_word in convert_to_original_length(dec_words) if d_word not in nltk_stopwords and len(d_word) > 2])

    for word in enc_words:
        if word in concept_dic.keys() and word not in nltk_stopwords and len(word) > 1:  
            # print('w',word)
            for d_word in (dec_words):
                for j, triples in enumerate(concept_dic[word]):
                    if d_word in triples[0]:
                        # print('ee',d_word, triples[0])
                        x.append([word, triples[1], d_word])
                        # print('pp',word, triples[1], d_word)
                        break

    return x



def convert_to_original_length(sentence):
    r = []
    r_tags = []

    for index, token in enumerate(sentence):
        if token.startswith("##"):
            if r:
                r[-1] = f"{r[-1]}{token[2:]}"
        else:
            r.append(token)
            # r_tags.append(tags[index])
    return r

# def return_kg(enc_words, dec_words):
#     NAF = ["_NAF_H", '_NAF_R', "_NAF_T"]
#     triples_list = []
#     triples_len = 20
#     entity = []
#     # print(enc_words)
#     # print(set(convert_to_original_length(dec_words)))

#     for word in enc_words:
#         if word in concept_dic.keys() and word not in nltk_stopwords and len(word) > 1:  
#             # print('w',word)
#             x = []
#             dec_words = set([d_word for d_word in convert_to_original_length(dec_words) if d_word not in nltk_stopwords and len(d_word) > 2])
#             for d_word in (dec_words):
#                 for j, triples in enumerate(concept_dic[word]):
#                     if d_word in triples[0]:
#                         # print('ee',d_word, triples[0])
#                         x.append([word, triples[1], d_word])
#                         entity.append(triples[0])
#                         # print('pp',word, triples[1], d_word)
#                         break

#             x = x[:20]
#             # print(x)
#             # print(len(x))
#             if len(x) == 0:
#                 for i, triples in enumerate(concept_dic[word]):
#                     x.append([word, triples[1], triples[0]])
#                     entity.append(triples[0])

#                     # print('ppp',word, triples[1], triples[0])
#                     if (i+1) == triples_len:
#                         break

#             x = x + [NAF]*(triples_len - len(x))
#             triples_list.append(x)
#         else:
#             triples_list.append([NAF]*triples_len)

#     entity = [item.split('_') if '_' in item else [item] for item in entity]
#     entity = [i for item in entity for i in item if i not in nltk_stopwords]
#     # print(set(entity))

#     return triples_list, list(set(entity))

def make_dataset(data, file_name='train_data.pth'):
    train_data = []
    count = 0
    for j,d in enumerate(data):
        print(j)
        d_len = len(d)
        for i in range(d_len // 2):
            # print('src', d[:2 * i + 1])
            # print('trg', d[2 * i + 1])

            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input, kg_enc_input = seq2token_ids(d[:2 * i + 1],
                                                                                                 d[2 * i + 1])
            train_data.append((encoder_input,
                               decoder_input,
                               mask_encoder_input,
                               mask_decoder_input, kg_enc_input))
        # if j == 100:
        #     break
        # count += 1

    encoder_input, \
    decoder_input, \
    mask_encoder_input, \
    mask_decoder_input, kg_enc_input = zip(*train_data)

    encoder_input = torch.stack(encoder_input)
    decoder_input = torch.stack(decoder_input)
    kg_enc_input = torch.stack(kg_enc_input)
    # kg_dec_input = torch.stack(kg_dec_input)

    # entity = torch.stack(entity)
    # print(entity.size())

    mask_encoder_input = torch.stack(mask_encoder_input)
    mask_decoder_input = torch.stack(mask_decoder_input)

    train_data = [encoder_input, decoder_input, mask_encoder_input, mask_decoder_input, kg_enc_input]

    print('encoder_input',encoder_input.size())
    print(kg_enc_input.size())

    torch.save(train_data, file_name)



def get_splited_data_by_file(dataset_file):

    datasets = [[], [], []]

    with open(dataset_file, "r", encoding='utf-8') as f:
        json_data = f.read()
        data = json.loads(json_data)


    for d in data[:]:
        lst = []
        dialogue_len = 0
        for x in d['Dialogue']:
            lst = x.split()
            dialogue_len += 1
            if len(lst) < 4:
                if dialogue_len == 2:
                    data.remove(d)
                    break
                # else:
                #     d['Dialogue'] = d['Dialogue'][:dialogue_len-2]

    total_id_num = len(data)
    validate_idx = int(float(total_id_num) * 8 / 10)
    test_idx = int(float(total_id_num) * 9 / 10)

    datasets[0] = [d['Dialogue'] for d in data[:validate_idx]]
    datasets[1] = [d['Dialogue'] for d in data[validate_idx:test_idx]]
    datasets[2] = [d['Dialogue'] for d in data[test_idx:]]

    # print(datasets)
    return datasets


# print(data[0][100])
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_size', default=400, type=int, required=False)
    parser.add_argument('--decoder_size', default=100, type=int, required=False)
    parser.add_argument('--HCM_datapath', default='../../Data/covid_data/HCM/', type=str, required=False)
    parser.add_argument('--Icliniq_datapath', default='../../Data/covid_data/Icliniq/', type=str, required=False)
    parser.add_argument('--json_datapath', default='../../Data/covid_data/json_files/', type=str, required=False)
    parser.add_argument('--save', default='../../preprocessed_data/kg_data_pointer/data/c_data_3', type=str, required=False)
    parser.add_argument('--path', default='../../triples_prep/v1/covid_kg.pkl', type=str, required=False)
    parser.add_argument('--entity_dic', default='../../triples_prep/covid/entity2id.txt', type=str, required=False)
    parser.add_argument('--rel_dic', default='../../triples_prep/covid/relation2id.txt', type=str, required=False)
    # parser.add_argument('--dump', default='Bert/raw/', type=str, required=False)
    # parser.add_argument('--data', default='../../triples_prep/covid_dialog/', type=str, required=False)

    args = parser.parse_args()

    # tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    tok = BertTokenizer.from_pretrained('bert-base-uncased', \
                                              never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[END]"))

    MAX_ENCODER_SIZE = args.encoder_size
    MAX_DECODER_SIZE = args.decoder_size


    print('loading kg')
    concept_dic = pickle.load(open(args.path,'rb'))

    print(len(list(concept_dic.keys())))
    print(list(concept_dic.keys())[:10])

    entity_list = []
    relation_list = []

    for item in concept_dic:
        entity_list.append(item)
        for i in concept_dic[item]:
            entity_list.append(('_').join(i[0].split()))
            relation_list.append(i[1])


    entity_list += ['_NAF_H', '_NAF_T']
    relation_list += ['_NAF_R']
    relation_list = set(relation_list)
    entity_list = set(entity_list)

    print('rr',relation_list)

    relation_dic = {}
    entity_dic = {}

    f = open(args.entity_dic,'w')
    f1 = open(args.rel_dic,'w')

    f.write(str(len(list(entity_list))))
    f1.write(str(len(list(relation_list))))

    f.write('\n')
    f1.write('\n')

    for i,item in enumerate(relation_list):
        relation_dic[item] = i
        f1.write(str(item) + '\t' + str(i))
        f1.write('\n')


    for i,item in enumerate(entity_list):
        entity_dic[item] = i
        f.write(str(item) + '\t' + str(i))
        f.write('\n')

    f.close()
    f1.close()
        
    # prepare vocab

    tok.save_vocabulary('./vocab/')

    f_v = open('vocab/vocab.txt')
    vocab = f_v.read().split('\n')
    print(vocab[-10:])


    entity_l = [item.split('_') if '_' in item else [item] for item in entity_list]
    entity_l = [i for item in entity_l for i in item if i not in nltk_stopwords]

    entity_list = set(list(entity_list) + list(set(entity_l)))


    new_entity_list = []

    for item in entity_list:
        if item in vocab:
            pass
        else:
            new_entity_list.append(item)

    new_vocab = vocab[:-1] + list(new_entity_list)
    print(new_vocab[:10])
    print(new_vocab[-10:])

    f_voc = open('vocab/new_vocab.txt','w')

    for i,item in enumerate(new_vocab):
        f_voc.write(str(item))
        f_voc.write('\n')
    f_voc.close()


    vocab_dic = {}
    n_voc = open('vocab/new_vocab.txt').read().split('\n')[:-1]

    for i,item in enumerate(n_voc):
        vocab_dic[item] = i

    print(len(vocab_dic))
    print(vocab_dic['[CLS]'])
    print(vocab_dic['[PAD]'])
    
    # tokenizer = BertTokenizer.from_pretrained('vocab/new_vocab.txt', never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[END]"))

    # print(tokenizer.convert_tokens_to_ids(['fibrosis']))
    # print(tokenizer.convert_tokens_to_ids(['_NAF_H']))
    # print(tokenizer.convert_tokens_to_ids(['[PAD]']))



    path3 = args.json_datapath
    total = 0

    tot_data = [[], [], []]



    for root,dirnames,filenames in os.walk(path3):
        for filename in filenames:
            json_file = os.path.join(os.path.join(path3,filename))
            temp = get_splited_data_by_file(json_file)
            tot_data[0].extend(temp[0])
            tot_data[1].extend(temp[1])
            tot_data[2].extend(temp[2])


    data = tot_data

    print(len(data[0]))
    print(len(data[1]))
    print(len(data[2]))

    print(f'Process the train dataset')
    make_dataset(data[0], args.save + '/train_data.pkl')

    print(f'Process the validate dataset')
    make_dataset(data[1], args.save + '/validate_data.pkl')

    print(f'Process the test dataset')
    make_dataset(data[2], args.save + '/test_data.pkl')



    f_voc = open('vocab/new_vocab.txt','w')

    for key in (vocab_dic):
        f_voc.write(str(vocab_dic[key]))
        f_voc.write('\n')
    f_voc.close()
    print(len(vocab_dic))