import torch
import numpy as np
import os
from tqdm import tqdm


def train(model, optimizer, data_loader, device):
    model.train()
    
    total = 0.0
    correct = 0.0
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
            
        ccounter += 1
        correct += outputs["train_correct"]
        total += outputs["train_total"]
            
    acc = float(correct) / total if total>0 else 0
    loss = float(total_loss) / ccounter
    
    return loss, acc
    
def val(model, data_loader, deivce):
    model.eval()
    
    
    total = 0.0
    correct = 0.0
    ccounter = 0.0
    total_loss = 0.0
    total_cer = 0.0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            outputs = model(batch, "val")
            
            loss = outputs["val_loss"]
            total_loss += loss
            ccounter += 1
            correct += outputs["val_correct"]
            total += outputs["val_total"]
            
            
        acc = float(correct) / total if total>0 else 0
        loss = float(total_loss) / ccounter

    return loss, acc

def trainer(num_epochs, model, loader_dict, optimizer, device, outdir, phasename, is_use_wandb=False):
	
	best_val_loss = 1000000000
	for epoch in range(num_epochs):
		print("Exp:{}, Epoch:{}".format(phasename, epoch+1))
		train_loss, train_acc = train(model, optimizer, loader_dict["train"], device)
		val_loss, val_acc = val(model, loader_dict["val"], device)

		print("Train loss: {}".format(train_loss))
		print("Train Acc: {}".format(train_acc))
		print("Val loss: {}".format(val_loss))
		print("Val Acc: {}".format(val_acc))
		#print("Val dialog f1: {}".format(val_dialog_f1))
		#print("Val system f1: {}".format(val_system_f1))
# 		if best_val_loss>val_loss:
# 			best_val_loss = val_loss
# 			torch.save(model.state_dict(), os.path.join(outdir, "best_val_loss_model.pth"))
		torch.save(model.state_dict(), os.path.join(outdir, "model_epoch{}_loss{:.3f}.pth".format(epoch+1, val_loss)))
		torch.save(model.acoustic_encoder.state_dict(), os.path.join(outdir, "acoustic_best_val_loss_model.pth"))
		torch.save(model.semantic_encoder.state_dict(), os.path.join(outdir, "semantic_best_val_loss_model.pth"))

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
