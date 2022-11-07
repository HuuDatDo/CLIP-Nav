import torch
import torch.nn as nn

from learning.modules.clipunet.extractor import Extractor


def train_extractor():
    epoch = 100
    dataset = "..."
    model = Extractor()
    criterion = nn.CrossEntropyLoss()
    optimizer = nn.optim.Adam(model.parameters(), lr= 0.0001)
    
    for j in range(epoch):
        for i, data in enumerate(dataset):
            instructions, nouns = data
            
            optimizer.zero_grad()
            
            pred = model(instructions)
            loss = criterion(pred, nouns)
            
            loss.backward()
            optimizer.step()
            
            if i%20 ==0:
                print(f"Epochs: {i+1} loss: {loss.item()}")
    
    print("Finish training")