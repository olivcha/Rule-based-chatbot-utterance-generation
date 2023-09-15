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
    empathy_model_2.load_state_dict(torch.load('../../empathy_output/best_model_first_ft.pt', map_location=torch.device(DEVICE)), strict=False) #change path
    empathy_model_2 = empathy_model_2.to(DEVICE)
    

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
    
    score_data_path = "data/new_utterances.csv"
    
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
    dataset = pd.read_csv(score_data_path,header=0, usecols=['empathetic_rewriting']).dropna()
    # dataset = pd.read_csv(score_data_path,header=0, usecols=['Response']).dropna()

    scores = []
    
    
    print("generating scores with 2ft model...")
    scores = []
    for row in dataset.itertuples():
        sentence = row[1]
        score = get_empathy(sentence, empathy_model_2)
        print("score: ", str(score))
        score_numeric = label2int[score]
        # print("score numeric: ", str(score_numeric))
        scores.append(score_numeric)
        

    # add the scores to the CSV file
    data = pd.read_csv("data/new.csv")
    data['empathy_score'] = pd.Series(scores)

    data.to_csv(score_data_path, index=False)