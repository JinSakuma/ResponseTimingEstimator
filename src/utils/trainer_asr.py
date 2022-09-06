import torch
import numpy as np
import os
from tqdm import tqdm
import wandb


def train(model, optimizer, data_loader, device):
    model.train()
    
    correct = 0.0
    ccounter = 0.0
    total_loss = 0.0
    for batch in tqdm(data_loader):
        optimizer.zero_grad()
        
        outputs = model(batch, "train")

        loss = outputs["train_loss"]
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.detach().cpu().numpy()
        ccounter += 1
        correct += outputs["train_cer"]
            
    cer = float(correct) / ccounter
    loss = float(total_loss) / ccounter
    
    return loss, cer
    
def val(model, data_loader, deivce):
    model.eval()
    
    correct = 0.0
    ccounter = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            outputs = model(batch, "val")
            total_loss += outputs["val_loss"].detach().cpu().numpy()
            correct += outputs["val_cer"]
            ccounter += 1
            
        cer = float(correct) / ccounter
        loss = float(total_loss) / ccounter

    return loss,  cer

def trainer(num_epochs, model, loader_dict, optimizer, device, outdir):

    best_val_loss = 1000000000
    for epoch in range(num_epochs):
        print("Epoch:{}".format(epoch+1))
        train_loss, train_cer = train(model, optimizer, loader_dict["train"], device)
        val_loss, val_cer = val(model, loader_dict["val"], device)

        print("Train loss: {}".format(train_loss))
        print("Train CER: {}".format(train_cer))
        print("Val loss: {}".format(val_loss))
        print("Val CER: {}".format(val_cer))
        if best_val_loss>val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(outdir, "best_val_loss_model.pth"))


