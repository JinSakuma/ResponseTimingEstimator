import torch
import numpy as np
import os
from tqdm import tqdm
#import wandb


def train(model, optimizer, data_loader, device):
    model.train()
    
    total_dialog = 0.0
    correct_dialog = 0.0
    total_system = 0.0
    correct_system = 0.0
    ccounter = 0.0
    total_loss = 0.0
    total_cer = 0.0
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        optimizer.zero_grad()
        
        outputs = model(batch, "train")

        loss = outputs["train_loss"]
        
        if type(loss)==torch.Tensor:
            loss.backward()
            loss = loss.detach().cpu().numpy()
        optimizer.step()
        
        total_loss += loss
            
        #total_P += outputs["val_dialog_acts_precision"]
        #total_R += outputs["val_dialog_acts_recall"]
        #total_F1 += outputs["val_dialog_acts_f1"]
        #total_cer += outputs["train_asr_cer"]
        ccounter += 1
        #correct_dialog += outputs["train_dialog_acts_acc"]
        #total_dialog += outputs["train_num_dialog_acts_total"]
        #correct_system += outputs["train_system_acts_acc"]
        #total_system += outputs["train_num_system_acts_total"]
            
    #precision = total_P / ccounter
    #recall = total_R / ccounter
    #f1 = total_F1 / ccounter
    #cer = total_cer / ccounter 
    #acc_dialog = float(correct_dialog) / total_dialog if total_dialog>0 else 0
    #acc_system = float(correct_system) / total_system if total_system>0 else 0
    
    loss = float(total_loss) / ccounter
    
    return loss#, cer
    
def val(model, data_loader, deivce):
    model.eval()
    
    total = {"asr_cer": 0, "loss":0,
             "dialog_p": 0, "dialog_r": 0, "dialog_f1": 0,
             "dialog_correct": 0, "dialog_total": 0,
             "system_p": 0, "system_r": 0, "system_f1": 0,
             "system_correct": 0, "system_total": 0,
            }
    ccounter = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            outputs = model(batch, "val")
            loss = outputs["val_loss"]
            if type(loss)==torch.Tensor:
                loss = loss.detach().cpu().numpy()
            total["loss"] += loss
            
            #total["dialog_p"] += outputs["val_dialog_acts_precision"]
            #total["dialog_r"] += outputs["val_dialog_acts_recall"]
            #total["dialog_f1"] += outputs["val_dialog_acts_f1"]
            #total["dialog_correct"] += outputs["val_dialog_acts_acc"]
            #total["dialog_total"] += outputs["val_num_dialog_acts_total"]
            #total["system_p"] += outputs["val_system_acts_precision"]
            #total["system_r"] += outputs["val_system_acts_recall"]
            #total["system_f1"] += outputs["val_system_acts_f1"]
            #total["system_correct"] += outputs["val_system_acts_acc"]
            #total["system_total"] += outputs["val_num_system_acts_total"]
            #total["asr_cer"] += outputs["val_asr_cer"]
            ccounter += 1
            
        #dialog_p = float(total["dialog_p"]) / ccounter
        #dialog_r = float(total["dialog_r"]) / ccounter
       	#dialog_f1 = float(total["dialog_f1"]) / ccounter
        #dialog_acc = float(total["dialog_correct"]) / total["dialog_total"] if total["dialog_total"]>0 else 0
        #system_p = float(total["system_p"]) / ccounter
        #system_r = float(total["system_r"]) / ccounter
       	#system_f1 = float(total["system_f1"]) / ccounter
        #system_acc = float(total["system_correct"]) / total["system_total"] if total["system_total"]>0 else 0
       	#cer = float(total["asr_cer"]) / ccounter 
        loss = float(total["loss"]) / ccounter

    return loss #, cer #, (dialog_p, dialog_r, dialog_f1, dialog_acc), (system_p, system_r, system_f1, system_acc)

def trainer(num_epochs, model, loader_dict, optimizer, device, outdir, is_use_wandb=False):

    best_val_loss = 1000000000
    for epoch in range(num_epochs):
        print("Epoch:{}".format(epoch+1))
        train_loss = train(model, optimizer, loader_dict["train"], device)
        val_loss = val(model, loader_dict["val"], device)
        #val_dialog_p, val_dialog_r, val_dialog_f1, val_dialog_acc = val_dialog
        #val_system_p, val_system_r, val_system_f1, val_system_acc = val_system

        print("Train loss: {}".format(train_loss))
        #print("Train WER: {}".format(train_cer))
        print("Val loss: {}".format(val_loss))
        #print("Val WER: {}".format(val_cer))
        #print("Val dialog f1: {}".format(val_dialog_f1))
        #print("Val system f1: {}".format(val_system_f1))
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(outdir, "best_val_loss_model.pth"))
            #torch.save(model.state_dict(), os.path.join(outdir, "model_epoch{}_loss{:.3f}.pth".format(epoch+1, val_loss)))
            #torch.save(model.asr_model.state_dict(), os.path.join(outdir, "asr_best_val_loss_model.pth"))

        with open(os.path.join(outdir, 'loss.txt'), 'a') as f:
            f.write(str(val_loss)+'\n')

# 		if is_use_wandb:
# 			wandb.log({
# 				"Train Loss": train_loss,
# 				"Train Dialog Acts Acc": train_dialog_acc,
# 				"Train System Acts Acc": train_system_acc,
# 				"Train WER": train_cer,
# 				"Val Loss": val_loss,
# 				"Val Dialog Acts Acc": val_dialog_acc,
# 				"Val Dialog Acts Precision": val_dialog_p,
# 				"Val Dialog Acta Recall": val_dialog_r,
# 				"Val Dialog Acts F1": val_dialog_f1,
# 				"Val System Acts Acc": val_system_acc,
# 				"Val System Acts Precision": val_system_p,
# 				"Val System Acts Recall": val_system_r,
# 				"Val System Acts F1": val_system_f1,
# 				"Val WER": val_cer,
# 			})
