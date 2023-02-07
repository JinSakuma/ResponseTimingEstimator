# coding: UTF-8

import os
import json
import random
import math
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotmap import DotMap
from tqdm import tqdm
import sys
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report
    

def turn_taking_evaluation(y_pred, y_true, threshold=0.5, frame=50):
    
    target = False
    pred = False
    flag = True
    fp_flag = False
    AB, C, D, E = 0, 0, 0, 0    
    pred_frame, target_frame = -1, -1
    #for i in range(1,len(y_pred)-1):
    for i in range(len(y_pred)-1):
                
        #  予測が閾値を超えたタイミング
        if y_pred[i] >= threshold and flag:
            pred = True
            flag = False
            pred_frame = i

        #  正解ラベルのタイミング
        if y_true[i] > 0:
            target = True
            target_frame = i

        
    flag = True
    if pred and target:
        AB += 1
    if (pred and not target) or fp_flag:
        C += 1
    if target and not pred:
        E += 1
    if not target and not pred:
        D += 1

    # TP, FP, FN, TN
    return AB, C, E, D, pred_frame*frame, target_frame*frame

# A：システムが「発話せよ」と判断したタイミングの周囲で，実際にウィザードが発話している数
# B：システムが「発話せよ」と判断したタイミングの外で，ウィーザードが発話した数
# C：システムが「発話せよ」としたにも関わらず，ウィーザードはどこでも発話しなかった数（他の人が話し始めた，規定時間以上経過した　など）
# D：システムは「発話せよ」と判断せず（＝発話するなと判断し），実際ウィザードも発話しなかった数
# E：システムは「発話せよ」と判断しなかったにも関わらず，ウィザードがどこかで発話した数
def timing_evaluation(y_pred, y_true, u_label, tt, threshold=0.5, frame=50):
    
    target = False
    pred = False
    flag = True
    fp_flag = False
    AB, C, D, E = 0, 0, 0, 0    
    pred_frame, target_frame = -10000, -10000
    for i in range(1, len(y_pred)-1):
                
        #  予測が閾値を超えたタイミング
        if y_pred[i] >= threshold and flag:
            if u_label[i]>0.5 and i<tt:
                fp_flag=True
            else:
                if y_pred[tt-1]<0.5:
                    pred = True
                    flag = False
                    pred_frame = i                

        #  正解ラベルのタイミング
        if y_true[i] > 0:
            target = True
            target_frame = i

        
    flag = True
    if pred and target:
        AB += 1
    if (pred and not target): # or fp_flag:
        C += 1
    if target and not pred:
        E += 1
    if not target and not pred:
        D += 1
        
    if fp_flag:
        C += 1
    else:
        D += 1


    # TP, FP, FN, TN
    return AB, C, E, D, pred_frame, target_frame
    
    
def tester(config, device, test_loader, model, model_dir, out_dir, resume_name, resume=True):
    
    epoch_list =  os.listdir(model_dir)
    min_loss = 1000000
    w_path = epoch_list[-1]
    for weight in epoch_list:        
        if "model" in weight:
            w_path = weight
            
    path = os.path.join(model_dir, w_path)
    model.load_state_dict(torch.load(path))#, strict=False)
    model.to(device)
    
    dic={"TP": 0, "TP_label": [], "TP_pred": [], "FN": 0, "FN_label": [], "FP": 0, "TN": 0}
    y_pred_list = []
    system_label_list = []
    y_label_list = []
    uttr_label_list = [] 

    y_pred_list_test = []
    offset_list_test = []
    silence_list_test = []
    system_label_list_test = []
    y_label_list_test = []
    uttr_label_list_test = [] 
    barge_in_list_test = [] 
    timing_loss = 0
    split='test'
    with torch.no_grad():
        for batch in tqdm(test_loader):
            chs = batch[0]
            texts = batch[1]
            kanas = batch[2]
            idxs = batch[3]
            vad = batch[4]
            #turn = batch[5].to(self.device)
            #last_ipu = batch[6].to(self.device)
            targets = batch[7].to(model.device)
            feats = batch[8].to(model.device)
            input_lengths = batch[9] #.to(self.device)
            offsets = batch[10] #.to(self.device)
            indices = batch[11] #.to(self.device)
            is_barge_in = batch[12] #.to(self.device)
            batch_size = int(len(chs))            

            embs = model.feature_extractor(feats, idxs, input_lengths, texts, indices, split)
            outputs = model.timing_model(embs, input_lengths)        

            loss, acc = 0, 0
            out_list, label_list, uttr_list, silence_list = [], [], [], []
            for i in range(batch_size):
                loss = loss+model.timing_model.get_loss(outputs[i][:input_lengths[i]], targets[i][:input_lengths[i]])
                #silence = self.timing_model.get_silence(vad_preds[i][:input_lengths[i]], indices[i], split)

                out_list.append(outputs[i][:input_lengths[i]])
                #silence_list.append(outputs[i][:input_lengths[i]])
                label_list.append(targets[i][:input_lengths[i]])
                uttr_list.append(vad[i][:input_lengths[i]])

            for idx in range(len(out_list)):
                y_pred = torch.sigmoid(out_list[idx]).detach().cpu().numpy()
                system = label_list[idx].detach().cpu().numpy()
                y_label = system[1:]-system[:-1]
                uttr = uttr_list[idx].detach().cpu().numpy()

                y_pred_list+=list(y_pred)
                system_label_list+=list(system)
                y_label_list+=list(y_label)
                uttr_label_list+=list(uttr)

                y_pred_list_test.append(y_pred)
                system_label_list_test.append(system)
                y_label_list_test.append(y_label)
                uttr_label_list_test.append(uttr)
                offset_list_test.append(offsets[idx])
                barge_in_list_test.append(is_barge_in[idx])
                #silence_list_test.append(silence_list[idx].detach().cpu().numpy())
                
                
    triggers = []
    ipu_labels = []
    eou_list = []
    S=-5 # 5
    for i in range(len(uttr_label_list_test)):
        eou = np.where(y_label_list_test[i]==1)[0][0]-(abs(offset_list_test[i])//50*offset_list_test[i]//abs(offset_list_test[i]))
        eou_list.append(eou)

        turn = np.zeros(len(y_label_list_test[i]))
        turn[:eou] = 1

        uu = 1-uttr_label_list_test[i]
        timing = np.where(y_label_list_test[i]==1)[0][0]
        t_ups = np.where(uu[1:]-uu[:-1]==-1)[0]+1
        if len(t_ups[t_ups<(eou-S)])>0:
            t_up = t_ups[t_ups<(eou-S)][-1]
        else:
            t_up = 0

        ipu_target = np.zeros(len(uttr_label_list_test[i]))
        ipu_target[:eou] = 1
        ipu_labels.append(ipu_target)
        triggers.append(t_up)

    dic_test={"TP": 0, "TP_label": [], "TP_pred": [], "FN": 0, "FN_label": [], "FP": 0, "TN": 0}
    dic_info={"target": [], "pred": [], "type": [], "barge_in": []}
    thres = 0.5
    frame_length = 50
    for i in tqdm(range(len(y_pred_list_test))):
        TP, FP, FN, TN, pred, target = timing_evaluation(y_pred_list_test[i], y_label_list_test[i], ipu_labels[i], triggers[i], threshold=thres)
        pred = (pred - eou_list[i])*frame_length
        target = (target - eou_list[i])*frame_length
        barge_in = barge_in_list_test[i]

        if TP>0:
            dic_test["TP"]+=1
            dic_info["target"].append(target)
            dic_info["pred"].append(pred)
            dic_info["type"].append(1)
            dic_info["barge_in"].append(barge_in)

        elif FN>0:
            dic_test["FN"]+=1
            dic_info["target"].append(target)
            dic_info["pred"].append(-1000000)
            dic_info["type"].append(0)
            dic_info["barge_in"].append(barge_in)
        if FP>0: 
            dic_test["FP"]+=FP
        if TN>0:
            dic_test["TN"]+=TN

    df_test = pd.DataFrame({
        'type': dic_info['type'], 
        'target': dic_info["target"],
        'pred': dic_info["pred"],
        'barge_in': dic_info["barge_in"],
    })

    df_test['error'] = df_test['target'].values - df_test['pred'].values
    
    
    # # A：システムが「発話せよ」と判断したタイミングの周囲で，実際にウィザードが発話している数
    # # B：システムが「発話せよ」と判断したタイミングの外で，ウィーザードが発話した数
    # # C：システムが「発話せよ」としたにも関わらず，ウィーザードはどこでも発話しなかった数（他の人が話し始めた，規定時間以上経過した　など）
    # # D：システムは「発話せよ」と判断せず（＝発話するなと判断し），実際ウィザードも発話しなかった数
    # # E：システムは「発話せよ」と判断しなかったにも関わらず，ウィザードがどこかで発話した数
    err_list = [250, 500, 1000]
    # err_list = [500, 1000, 1500]
    for err in err_list:

        df = df_test
        df_TP = df[df['type']==1]

        A = len(df_TP[abs(df_TP['error'])<=err])
        B = len(df_TP[abs(df_TP['error'])>err])
        C = dic_test['FP']
        D = dic_test['TN']
        E = dic_test['FN']

        recall = A / (A+B+E) if A+B+E>0 else 0
        precision = A / (A+B+C) if (A+B+C)>0 else 0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision)>0 else 0

        mae = np.array([abs(e) for e in df_TP[abs(df_TP['error'])<=err]['error'].values]).mean()

        print(A, B, C, D, E)
        print("許容誤差{}ms - precision: {:.3f}, recall: {:.3f}, f1: {:.3f}, MAE: {:.1f}".format(err, precision, recall, f1, mae))
        print()
        
    if resume:
        df_tp = df_test[df_test['type']==1]
        
        # 図用のnpy保存
        name = '{}'.format(resume_name)

        err_list = np.arange(0, 10050, 50)
        score_dict = {'precision': [], 'recall': [], 'f1': []}

        os.makedirs(os.path.join(out_dir, 'npy'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'csv'), exist_ok=True)
        for err in err_list:

            df = df_test
            df_TP = df[df['type']==1]

            A = len(df_TP[abs(df_TP['error'])<=err])
            B = len(df_TP[abs(df_TP['error'])>err])
            C = dic_test['FP']
            D = dic_test['TN']
            E = dic_test['FN']

            recall = A / (A+B+E) if A+B+E>0 else 0
            precision = A / (A+B+C) if (A+B+C)>0 else 0
            f1 = 2 * recall * precision / (recall + precision) if (recall + precision)>0 else 0

            score_dict['recall'].append(recall)
            score_dict['precision'].append(precision)
            score_dict['f1'].append(f1)

        np.save(os.path.join(out_dir, 'npy', 'recall_{}.npy'.format(name)), np.asarray(score_dict['recall']))
        np.save(os.path.join(out_dir, 'npy','precision_{}.npy'.format(name)), np.asarray(score_dict['precision']))
        np.save(os.path.join(out_dir, 'npy','f1_{}.npy'.format(name)), np.asarray(score_dict['f1']))


        errs = np.array([abs(e) for e in df_tp['error'].values])
        np.save(os.path.join(out_dir, 'npy', 'errors_{}.npy'.format(name)), errs)
        df_test.to_csv(os.path.join(out_dir, 'csv', 'df_{}.csv'.format(name)))
        
    labels_test = []
    predictions_test = []

    results = []
    silences = []
    cnt = 0
    for i in tqdm(range(len(y_pred_list_test))):

        res1=y_label_list_test[i][:-1]-y_label_list_test[i][1:]
        timing = np.where(res1==-1)[0][0]+1

        res=uttr_label_list_test[i][:-1]-uttr_label_list_test[i][1:]
        starts = np.where(res==-1)[0]+1
        ends = np.where(res==1)[0]+1

        if len(ends)>0:
            ends = ends[ends<=timing-offset_list_test[i]//50+1]
        elif offset_list_test[i]<=0:
            ends = np.array([len(y_pred_list_test[i])-1])
        else:        
            starts = [0]
            ends = [len(uu)]
            # raise NotImplemented
            tmp.append(i)

        if uttr_label_list_test[i][0]>0:
            starts = np.array([0]+starts.tolist())

        for j in range(len(ends)):

            if j < len(ends)-1:
                start = starts[j]
                end = starts[j+1]

                silence = starts[j+1] - ends[j]
                AB, C, E, D, pred, target = turn_taking_evaluation(y_pred_list_test[i][start:end], y_label_list_test[i][start:end], threshold=0.5)
            else:
                start = starts[j]
    #             if len(ends) < len(starts):
    #                 cnt += 1
    #                 continue

                silence = len(y_label_list_test[i]) - ends[j]
                AB, C, E, D, pred, target = turn_taking_evaluation(y_pred_list_test[i][start:], y_label_list_test[i][start:], threshold=0.5)
#                 if C or D:
#                     print(timing)
#                     print(starts)
#                     print(ends)
#                     print(AB, C, E, D, i, start)
#                     break

            assert AB+C+E+D<=1, 'evaluation error'

            if AB:
                out = 'TP'
                labels_test.append(1)
                predictions_test.append(1)
            elif C:
                out = 'FP'
                labels_test.append(0)
                predictions_test.append(1)
            elif D:
                out = 'TN'
                labels_test.append(0)
                predictions_test.append(0)
            elif E:
                out = 'FN'
                labels_test.append(1)
                predictions_test.append(0)
            else:
                NotImplemented

            results.append(out)
            silences.append(silence)
            
    df_silence = pd.DataFrame({'type': results, 'silence': silences})
    if resume:
        df_silence.to_csv(os.path.join(out_dir, 'csv', 'df_silence_{}.csv'.format(name)))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.path', help='path to config file')
    parser.add_argument('--weight', type=str, 
                        default='exp/timing/M1_3000_500/delay200/model_w_lm0714_n_best3_lr3_char_s12345/model_epoch18_loss128.960.pth',
                        help='path to model weight')
    parser.add_argument('--model', type=str, 
                        default='proposed',
                        help='model type: proposed or baseline')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--gpuid', type=int, default=0, help='gpu device id')
    parser.add_argument('--resume', type=bool, default=False, help='save npy file and dataframe or not')
    parser.add_argument('--name', type=str, default=None, help='name of save files')
    args = parser.parse_args()
          
    run(args)
    
    
    
