import torch
import numpy as np
import os
from tqdm import tqdm
import wandb


def train(model, optimizer, data_loader, device):
    model.train()
    
    correct_vad = 0.0
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
        correct_vad += outputs["train_vad_acc"]
            
    acc_vad = float(correct_vad) / ccounter
    loss = float(total_loss) / ccounter
    
    return loss, acc_vad
    
def val(model, data_loader, deivce):
    model.eval()
    
    total = {"loss":0, "vad_correct": 0}
    ccounter = 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            outputs = model(batch, "val")
            total["loss"] += outputs["val_loss"].detach().cpu().numpy()
            total["vad_correct"] += outputs["val_vad_acc"]
            ccounter += 1
            
        vad_acc = float(total["vad_correct"]) / ccounter
        loss = float(total["loss"]) / ccounter

    return loss,  vad_acc

def trainer(num_epochs, model, loader_dict, optimizer, device, outdir):

    best_val_loss = 1000000000
    for epoch in range(num_epochs):
        print("Epoch:{}".format(epoch+1))
        train_loss, train_vad_acc = train(model, optimizer, loader_dict["train"], device)
        val_loss, val_vad_acc = val(model, loader_dict["val"], device)

        print("Train loss: {}".format(train_loss))
        print("Train VAD: {}".format(train_vad_acc))
        print("Val loss: {}".format(val_loss))
        print("Val VAD: {}".format(val_vad_acc))
        if best_val_loss>val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(outdir, "best_val_loss_model.pth"))


