from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch
import logging
import os
from torch import nn
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from trainer import Trainer
from model import EmpathyClassificationModel
from utils import EmpathyDataset, TokenizersCollateFn, labels, label2int


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_empathy_score(model, tokenizer, sentence):
    inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        print('generating output for sentence...')
        outputs = model((input_ids, attention_mask))
        logits = outputs[0]
        empathy_score = torch.sigmoid(logits[0][1].item())
    
    return empathy_score


# #load emotion classifier (RoBERTa)
with torch.no_grad():
#     # empathy_model = EmpathyClassificationModel(AutoModelWithLMHead.from_pretrained("roberta-base").base_model, len(labels))
#     # empathy_model.load_state_dict(torch.load('second_try/best_model_first_ft.pt', map_location=torch.device(DEVICE)), strict=False) #change path
    
    empathy_model_2 = EmpathyClassificationModel(AutoModelWithLMHead.from_pretrained("roberta-base").base_model, len(labels))
    empathy_model_2.load_state_dict(torch.load('second_try/best_model_second_ft.pt', map_location=torch.device(DEVICE)), strict=False) #change path
    empathy_model_2 = empathy_model_2.to(DEVICE)
    
    empathy_model_3 = EmpathyClassificationModel(AutoModelWithLMHead.from_pretrained("roberta-base").base_model, len(labels))
    empathy_model_3.load_state_dict(torch.load('fourth_try/best_model_second_ft_1_triple.pt', map_location=torch.device(DEVICE)), strict=False) #change path
    empathy_model_3 = empathy_model_3.to(DEVICE)

def get_empathy(text, model): # roberta
    '''
    Classifies and returns the empathy level of a text string
    '''
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    t = ByteLevelBPETokenizer(
                "tokenizer/vocab.json", #change path
                "tokenizer/merges.txt"  #change path
            )
    t._tokenizer.post_processor = BertProcessing(
                ("</s>", t.token_to_id("</s>")),
                ("<s>", t.token_to_id("<s>")),
            )
    t.enable_truncation(512)
    t.enable_padding(pad_id=t.token_to_id("<pad>"))
    
    tokenizer = t
    encoded = tokenizer.encode(text)
    sequence_padded = torch.tensor(encoded.ids).unsqueeze(0).to(DEVICE)
    attention_mask_padded = torch.tensor(encoded.attention_mask).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model((sequence_padded, attention_mask_padded)).to(DEVICE)
        
    # top_p, top_class = output.topk(1, dim=1)
    # label = int(top_class[0][0])
    label = torch.argmax(output, dim=1).item()
    label_map = {v: k for k, v in label2int.items()}
    return label_map[label]



if __name__ == "__main__":
    if torch.cuda.is_available():
        # device = torch.device("cuda")
        print('Using GPU: {}'.format(torch.cuda.get_device_name(0)))
    else: 
        # device = torch.device("cpu")
        print('using cpu')
        logging.info('Using CPU')
    
    score_data_path = "data/new_utterances (3).csv"
    
    # tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    # model = AutoModelWithLMHead.from_pretrained("roberta-base")
    
    # # load the model and tokenizer
    # model_path = 'best_model_second_ft.pt'
    # tokenizer_collate_fn = TokenizersCollateFn()
    
    # # tokenizer = ByteLevelBPETokenizer.from_file("tokenizer/vocab.json", "tokenizer/merges.txt")
    # model = EmpathyClassificationModel(model.base_model, len(labels))
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # model.eval().to(device)
    
    # load the dataset to generate scores
    dataset = pd.read_csv(score_data_path,header=0, usecols=['empathetic_rewriting_new']).dropna()
    # dataset = pd.read_csv(score_data_path,header=0, usecols=['Response']).dropna()

    scores_3 = []
    
    # for i in dataset.itertuples():
    #     print(i[1])
    #     # print(j)
    #     print(type(i))
    #     # print(type(j))

    # sentence = "It's great that you recognize the importance of curiosity when it comes to exploring and discovering new things. Staying curious is the key to staying creative. I'm glad you know that curiosity can be practiced too! Let me know if you need any help or support in pursuing your passion and staying curious. I'm always here for you!"
    # score_3 = get_empathy(sentence, empathy_model_3)
    # score_2 = get_empathy(sentence, empathy_model_2)
    
    # print(sentence)
    # print("score_3: ", str(score_3))
    # print("score_2: ", str(score_2))
    
    print("generating scores with 3ft model...")
    for row in dataset.itertuples():
        sentence = row[1]
        # print(sentence)
        score = get_empathy(sentence, empathy_model_3)
        print("score (3ft): ", str(score))
        score_numeric = label2int[score]
        # print("score numeric: ", str(score_numeric))
        scores_3.append(score_numeric)
        
    # add the scores to the CSV file
    data = pd.read_csv(score_data_path)
    data['NEWempathy_score_triple'] = pd.Series(scores_3)
    data.to_csv(score_data_path, index=False)
    
    # add the scores to the CSV file
    # data = pd.read_csv(score_data_path)
    # data['empathy_score_1FT_sentence'] = pd.Series(scores_1)
    
    print("generating scores with 2ft model...")
    scores_2 = []
    for row in dataset.itertuples():
        sentence = row[1]
        score = get_empathy(sentence, empathy_model_3)
        print("score: ", str(score))
        score_numeric = label2int[score]
        # print("score numeric: ", str(score_numeric))
        scores_2.append(score_numeric)
        
    # # # add the scores to the CSV file
    # data = pd.read_csv(score_data_path)
    # data['empathy_score'] = scores
    # data.to_csv(score_data_path, index=False)
    
    # add the scores to the CSV file
    data = pd.read_csv(score_data_path)
    data['NEWempathy_score_2'] = pd.Series(scores_2)

    data.to_csv(score_data_path, index=False)