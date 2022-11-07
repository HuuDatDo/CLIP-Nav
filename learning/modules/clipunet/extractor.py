import torch
import transformers
import torch.nn as nn

from transformers import DistilBertTokenizer, DistilBertModel

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.head = nn.Sequential(
            nn.Linear(768,1),
            nn.Sigmoid())
        #Experiment on the threshold 
        self.threshold = 0.7
        
    def extract(self, instructions):
        x = self.tokenizer(instructions)
        x = self.model(x)
        x = self.head(x)
        x = x.squeeze()[1:-1]
        
        instructions = instructions.split(" ")
        nouns = []
        
        for i,u in enumerate(x):
            if u > self.threshold:
                nouns.append(instructions[i])
        
        return nouns
    
    #This function is for fine tuning the model, uncomment the return extract to extract it
    def forward(self, instructions):
        # return self.extract(instructions)
        x = self.tokenizer(instructions)
        x = self.model(x)
        x = self.head(x)
        x = x.squeeze()[1:-1]
        
        return x