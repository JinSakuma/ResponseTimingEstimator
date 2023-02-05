# coding: UTF-8

import os
import json
import random
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotmap import DotMap
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report
import sys
from src.utils.utils import load_config


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def turn_taking_evaluation(y_pred, y_true, threshold=0.5, frame=50):
    
    target = False
    pred = False
    flag = True
    fp_flag = False
    AB, C, D, E = 0, 0, 0, 0    
    pred_frame, target_frame = -1, -1
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
    
    epoch_list =  sorted(os.listdir(model_dir))
    min_loss = 1000000
    w_path = ''
    for weight in epoch_list[1:-1]:
        split_list = weight.split('_')
        if len(split_list)<3:
            continue 

        loss = float(split_list[-1].replace('loss', '').replace('.pth', ''))
        if min_loss > loss:
            w_path = weight
            min_loss = loss

    path = os.path.join(model_dir, w_path)
    
    model.load_state_dict(torch.load(path))#, strict=False)
    model.to(device)
        
    dic={"TP": 0, "TP_label": [], "TP_pred": [], "FN": 0, "FN_label": [], "FP": 0, "TN": 0}

    labels_test_list = []
    predictions_test_list = []

    y_label_list_test = []
    uttr_label_list_test = [] 
    offset_list_test = []
    barge_in_list_test = []

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            chs = batch[0]
            texts = batch[1]

            vads = batch[4]
            targets = batch[7].to(model.device)
            feats = batch[8].to(model.device)
            input_lengths = batch[9] #.to(model.device)
            offsets = batch[10]
            indices = batch[11]
            batch_size = int(len(chs))


            r_a_tmp = model.acoustic_encoder(feats, input_lengths)

            r_a_list = []
            targets_list = []
            text_list = []
            ends_list = []
            intervals_list = []
            label_list, uttr_list = [], []
            for i in range(len(chs)):

                system = targets[i][:input_lengths[i]].detach().cpu().numpy()
                y_label = system[1:]-system[:-1]
                uttr = vads[i][:input_lengths[i]].detach().cpu().numpy()

                y_label_list_test.append(y_label)
                uttr_label_list_test.append(uttr)
                offset_list_test.append(offsets[i])
    #             barge_in_list_test.append(barge_in[i])

                res_timing=(targets[i][:input_lengths[i]][:-1]-targets[i][:input_lengths[i]][1:]).detach().cpu().numpy()
                timing = np.where(res_timing==-1)[0][0]+1

                res=(vads[i][:input_lengths[i]][:-1]-vads[i][:input_lengths[i]][1:]).detach().cpu().numpy()
                starts = np.where(res==-1)[0]+1
                ends = np.where(res==1)[0]+1

                if len(ends)>0:
                    ends = ends[ends<=timing-offsets[i]//50+1]
                elif offsets[i]<=0:
                    ends = np.array([input_lengths[i]-1])
                else:
                    raise NotImplemented

                if len(ends)==0:
                    ends = np.array([input_lengths[i]-1])
                    ends_ = np.array([input_lengths[i]-1])
                else:
                    ends_ = ends.copy()
                    for j in range(len(ends)):                   
                        if ends[j]+4 < input_lengths[i]:
                            ends_[j] = ends[j] + 4    

                # ends += 4
                # print(timing)
                # print(starts)
                # print(ends)
                tmp = targets[i][ends]
                try:
                    tmp[:-1] = 0
                    tmp[-1] = 1
                except:
                    ends = np.where(res==1)[0]+1
                    print(ends)
                    print(timing)
                    print(offsets[i]//50+1)
                    print(aaa)


                r_a_list.append(r_a_tmp[i][ends])
                targets_list.append(tmp)
                text_list += np.array(texts[i])[ends_].tolist()
                ends_list += ends.tolist()
                interval = [0]+list(ends[1:]-ends[:-1])
                intervals_list+=interval

            r_a = torch.cat(r_a_list)
            targets_ = torch.cat(targets_list)
            tlen = [len(t) for t in text_list]

            tl_pre = 0
            end_pre = 0
            speaking_rates = []
            for j in range(len(tlen)):
                speaking_rates.append((tlen[j]-tl_pre)/ends_list[j]-end_pre)
                tl_pre = tlen[j]
                end_pre = ends_list[j]

            feat_t = torch.tensor([ends_list, tlen, intervals_list, speaking_rates]).permute(1,0).float().to(model.device)
            #feat_t = torch.tensor([ends_list, tlen, intervals_list]).permute(1,0).float().to(model.device)

            r_s = model.semantic_encoder(text_list)
            r_t = model.timing_encoder(feat_t)

            probs = model.gfblock(r_a, r_s, r_t)
            loss = model.gfblock.get_loss(probs, targets_)

            correct +=(torch.argmax(probs, dim=1)==targets_).sum().detach().cpu().numpy()
            total += len(targets_)

            labels_test_list.append(targets_.tolist())
            predictions_test_list.append(torch.argmax(probs, dim=1).tolist())
                
                
    timing_list = []
    eou_list = []
    for i in range(len(uttr_label_list_test)):
        eou = np.where(y_label_list_test[i]==1)[0][0]-(abs(offset_list_test[i])//50*offset_list_test[i]//abs(offset_list_test[i]))
        eou_list.append(eou)

        uu = uttr_label_list_test[i]
        timing = np.where(y_label_list_test[i]==1)[0][0]
        timing_list.append(timing)    

    # A：システムが「発話せよ」と判断したタイミングの周囲で，実際にウィザードが発話している数
    # B：システムが「発話せよ」と判断したタイミングの外で，ウィーザードが発話した数
    # C：システムが「発話せよ」としたにも関わらず，ウィーザードはどこでも発話しなかった数（他の人が話し始めた，規定時間以上経過した　など）
    # D：システムは「発話せよ」と判断せず（＝発話するなと判断し），実際ウィザードも発話しなかった数
    # E：システムは「発話せよ」と判断しなかったにも関わらず，ウィザードがどこかで発話した数

    # AB: TP, C: FP, D: TN, E: FN

    dic_test={"TP": 0, "TP_label": [], "TP_pred": [], "FN": 0, "FN_label": [], "FP": 0, "TN": 0}
    dic_info={"target": [], "pred": [], "type": [], "barge_in": []}
    thres = 0.5
    frame_length = 50
    for i in tqdm(range(len(y_label_list_test))):
        target = (timing_list[i] - eou_list[i])*frame_length
        pred = 350#200#300
    #     barge_in = barge_in_list_test[i]

        flg = False
        for j in range(len(labels_test_list[i])):

            if j < len(labels_test_list[i])-1:            
                if labels_test_list[i][j]==0 and predictions_test_list[i][j]==1: # FP
                    flg = True

            else:
                assert labels_test_list[j], "error"

                if flg:
                    dic_test["FP"]+=1
                    flg=False
                else:
                    dic_test["TN"]+=1

                if labels_test_list[i][j] and not predictions_test_list[i][j]:
                    dic_test["FN"]+=1
                    dic_info["target"].append(target)
                    dic_info["pred"].append(-1000000)
                    dic_info["type"].append(0)
    #                 dic_info["barge_in"].append(barge_in)
                else:
                    dic_test["TP"]+=1
                    dic_info["target"].append(target)
                    dic_info["pred"].append(pred)
                    dic_info["type"].append(1)
    #                 dic_info["barge_in"].append(barge_in)


    #type_list = [1 for i in range(dic_test["TP"])] + [0 for i in range(dic_test["FN"])]
    df_test = pd.DataFrame({
        'type': dic_info['type'], 
        'target': dic_info["target"],
        'pred': dic_info["pred"],
    #     'barge_in': dic_info["barge_in"],
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
        score_dict = {'precision': [], 'recall': [], 'f1': [],
                      'acc1': [], 'acc2': [], 'acc3': []
                     }

        #out_dir = 'exp/timing1128_4'
        #out_dir = 'exp/timing_cbs-t/M1_3000_500'
        os.makedirs(os.path.join(out_dir, 'npy2'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'csv2'), exist_ok=True)
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

        np.save(os.path.join(out_dir, 'npy2', 'recall_{}.npy'.format(name)), np.asarray(score_dict['recall']))
        np.save(os.path.join(out_dir, 'npy2','precision_{}.npy'.format(name)), np.asarray(score_dict['precision']))
        np.save(os.path.join(out_dir, 'npy2','f1_{}.npy'.format(name)), np.asarray(score_dict['f1']))


        errs = np.array([abs(e) for e in df_tp['error'].values])
        np.save(os.path.join(out_dir, 'npy2', 'errors_{}.npy'.format(name)), errs)
        df_test.to_csv(os.path.join(out_dir, 'csv2', 'df_{}.csv'.format(name)))
        
    labels_test = []
    predictions_test = []

    results = []
    silences = []
    cnt = 0
    for i in tqdm(range(len(y_label_list_test))):

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
                AB, C, E, D, pred, target = turn_taking_evaluation(predictions_test_list[i][start:end], y_label_list_test[i][start:end], threshold=0.5)
            else:
                start = starts[j]
    #             if len(ends) < len(starts):
    #                 cnt += 1
    #                 continue

                silence = len(y_label_list_test[i]) - ends[j]
                AB, C, E, D, pred, target = turn_taking_evaluation(predictions_test_list[i][start:], y_label_list_test[i][start:], threshold=0.5)
                if C or D:
                    break

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
        df_silence.to_csv(os.path.join(out_dir, 'csv2', 'df_silence_{}.csv'.format(name)))
        

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
    
    
    
