from transformers import AutoModelWithLMHead, AutoTokenizer
from torch import nn
from utils import Mish
import torch
import gc
import torch.nn.functional as F
import numpy as np


#load tokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
#load actual model, pass the number of labels etc.
model = AutoModelWithLMHead.from_pretrained('roberta-base')
base_model = model.base_model

#load pretrained tokenizer information
tokenizer.save_pretrained("tokenizer")


#note: the following code is partly adapted from Marcin Zablocki's tutorial 'custom classifier on top of bert-like language model'
#define an EmpathyClassificationModel class to do the actual fine-tuning

class EmpathyClassificationModel(nn.Module):
    ''' Definition of the finetuned LM, using the base model, cross entropy loss,
        output size 768, and dropout as a regularisation method.
    '''

    def __init__(self, base_model, n_classes, base_model_output_size=768, dropout=0.05):
        super().__init__()
        self.base_model = base_model
        
        self.base_model.requires_grad = True # allow updating params of the model
        # self.loss = nn.CrossEntropyLoss() #cross entropy loss since this is multi-class classification
        # self.loss.require_grad = True
        self.dropout = dropout
        self.base_model_output_size = base_model_output_size
        self.n_classes = n_classes

        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.base_model_output_size, self.base_model_output_size),
            Mish(),
            nn.Dropout(self.dropout),
            nn.Linear(self.base_model_output_size, self.n_classes)
        )

        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=0.02)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, input_, *args):
        X, attention_mask = input_
        hidden_states = self.base_model(X, attention_mask=attention_mask)
        return self.classifier(hidden_states[0][:, 0, :])