import torch
import numpy as np
import os
from tqdm import tqdm


def train(model, optimizer, data_loader, device):
    model.train()
    
    tpr = 0.0
    tnr = 0.0
    total = 0.0
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
        tpr += outputs["train_tpr"]
        tnr += outputs["train_tnr"]
        total += outputs["train_cnt"]
            
    tpr = float(tpr) / total
    tnr = float(tnr) / total
    loss = float(total_loss) / ccounter
    
    return loss, tpr, tnr
    
def val(model, data_loader, deivce):
    model.eval()
    
    tpr = 0.0
    tnr = 0.0
    total = 0.0
    correct = 0.0
    ccounter = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            outputs = model(batch, "val")
            total_loss += outputs["val_loss"].detach().cpu().numpy()
#             correct += outputs["val_acc"]
            ccounter += 1
            tpr += outputs["val_tpr"]
            tnr += outputs["val_tnr"]
            total += outputs["val_cnt"]
            
        #acc = float(correct) / ccounter
        tpr = float(tpr) / total
        tnr = float(tnr) / total
        loss = float(total_loss) / ccounter

    return loss, tpr, tnr

def trainer(num_epochs, model, loader_dict, optimizer, device, outdir, phasename):

    best_val_loss = 1000000000
    best_val_bacc = 0
    for epoch in range(num_epochs):
        print("Exp:{}, Epoch:{}".format(phasename, epoch+1))
        train_loss, train_tpr, train_tnr = train(model, optimizer, loader_dict["train"], device)
        val_loss, val_tpr, val_tnr = val(model, loader_dict["val"], device)

        print("Train loss: {}".format(train_loss))
        print("Train TPR: {}".format(train_tpr))
        print("Train TNR: {}".format(train_tnr))
        print("Val loss: {}".format(val_loss))
        print("Val TPR: {}".format(val_tpr))
        print("Val TNR: {}".format(val_tnr))
        val_bacc = (val_tpr+val_tnr)/2
        if best_val_bacc<val_bacc:
            best_val_bacc = val_bacc
            torch.save(model.state_dict(), os.path.join(outdir, "best_val_bacc_model.pth"))


