import json
import wave
import argparse 
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm


def check_data_nums(dic):    
    print(list(dic.keys()))
    print([len(dic[c]) for c in list(dic.keys())])    

def get_df(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    
    columns_agent = ['start', 'end', 'turn_id_agent', 'dialog_acts_agent', 'manual_dialog_acts_agent']
    columns_user = ['start', 'end', 'turn_id_user', 'dialog_acts_user', 'manual_dialog_acts_user'] 

    data_agent = {}
    data_user = {}
    for col in columns_agent:
        data_agent[col.replace('_agent', '')] = []

    for col in columns_user:
        data_user[col.replace('_user', '')] = []

    for i, line in enumerate(lines):
    #     if len(line.split())<4:
    #         print(i, line)
    #         break

        splits = line.split()
        col, start, end = splits[:3]
        if len(splits)<=3:
            val = ''
        else:
            if 'id' in col:
                val = int(splits[3])
            else:
                val = splits[3]

        if col in columns_agent:
            if col == columns_agent[2]:
                data_agent['start'].append(int(start))
                data_agent['end'].append(int(end))

            data_agent[col.replace('_agent', '')].append(val)
        elif col in columns_user:
            if col == columns_user[2]:
                data_user['start'].append(int(start))
                data_user['end'].append(int(end))

            data_user[col.replace('_user', '')].append(val)    
        else:
            pass
        
    if len(data_agent['manual_dialog_acts'])==0:
        data_agent['manual_dialog_acts'] = ['']*len(data_agent['dialog_acts'])
    elif len(data_agent['dialog_acts'])==0:
        data_agent['dialog_acts'] = ['']*len(data_agent['manual_dialog_acts'])
        
    if len(data_user['manual_dialog_acts'])==0:        
        data_user['manual_dialog_acts'] = ['']*len(data_user['dialog_acts'])
    elif len(data_user['dialog_acts'])==0:
        data_user['dialog_acts'] = ['']*len(data_user['manual_dialog_acts'])        

    try:
        df_agent = pd.DataFrame(data_agent)
        df_agent['spk'] = [0]*len(df_agent)
    except:
        return -1
    
    try:
        df_user = pd.DataFrame(data_user)
        df_user['spk'] = [1]*len(df_user)
    except:
        return -1

    df = pd.concat([df_agent, df_user])
    df = df.sort_values('start')
    
    return df

def check_da(da):
    da_list = []
    for split in da.split():
        if '(' not in split:
            da_list.append(split)
    return ''.join(da_list)

def select_da_label(df, path):
    labels = []
    cnt1 = 0
    cnt2 = 0
    for i in range(len(df['manual_dialog_acts'].tolist())):
        mda = check_da(df['manual_dialog_acts'].iloc[i])
        da = check_da(df['dialog_acts'].iloc[i])
        if mda!='' and da!='':
            if mda != da:                
                cnt1 += 1
            labels.append(mda)
        elif mda!='':
                labels.append(mda)
        elif da!='':
            labels.append(da)
        else:
            labels.append(da)
            cnt2 += 1
            
    return labels, cnt1, cnt2

def process_turn_id(df):
    turn_ids = []
    ids = df['turn_id'].tolist()
    cnt = 0
    pre = -100
    for j, i in enumerate(ids):
        if type(i)==str:
            turn_ids.append(''.join(i.split()))
            cnt += 1
            print(df['start'].iloc[j], df['turn_id'].iloc[j])
        else:
            turn_ids.append(i)
     
    return turn_ids, cnt

def modify_turn_id(df):
    turn_ids = df['turn_id'].tolist()    
    spks = df['spk'].tolist()
    
    pre_a = 11111111
    pre_u = 111111111
    new_ids = []
    for i in range(len(turn_ids)):
        if spks[i]==0:
            if turn_ids[i] == -1:
                new_ids.append(-1)
            elif pre_a == turn_ids[i]:
                new_ids.append(1)
            else:
                new_ids.append(0)
                
            pre_a = turn_ids[i]
        elif spks[i]==1:
            if turn_ids[i] == -1:
                new_ids.append(-1)
            elif pre_u == turn_ids[i]:
                new_ids.append(1)
            else:
                new_ids.append(0)
                
            pre_u = turn_ids[i]
        else:
            print('speaker error')
                
    return new_ids, turn_ids

def check_concat_da(df):
    das = df['da'].tolist()
    ids = df['turn_id'].tolist()
    new_ids = ids.copy()
    for i in range(len(das)):        
        if '+f' in das[i] and ids[i+1]==0:
            new_ids[i+1]=1        
        elif '+b' in das[i] and ids[i]==0:
            new_ids[i]=1
            
        if das[i]=='pf' and ids[i+1]==0:
            new_ids[i+1]=1
        elif das[i]=='pb' and ids[i]==0:
            new_ids[i]=1
        
        if das[i]=='eom' and ids[i]==0:
            new_ids[i]=1

    return new_ids

def make_turn_df(df, spk_id):
    
    df = df[df['turn_id']!=-1]
    
    turns = df['turn_id'].values
    num_turn = len(turns[turns==0])
    tid = -1
    
    info = {'start': [0]*num_turn, 'end': [0]*num_turn, 'spk':[spk_id]*num_turn, 'da': ['']*num_turn}    
    for i in range(len(df)):
        assert (i == 0 and turns[i]==0) or i > 0, 'turn start error, {}, {}'.format(i, turns[i])
        assert tid < num_turn, 'turn num error'
            
        if turns[i]==0:
            tid += 1
            info['start'][tid] = df['start'].iloc[i]
            info['end'][tid] = df['end'].iloc[i]
            info['da'][tid] +=  df['da'].iloc[i]
        else:
            info['end'][tid] = df['end'].iloc[i]
            info['da'][tid] +=  '/'+df['da'].iloc[i]
            
    return pd.DataFrame(info)

def get_offset(df):        
    offsets = [-1]
    changes = [0]
    pre = df['spk'].iloc[0]
    for i in range(1, len(df)):        
        offset = df['start'].iloc[i] - df['end'].iloc[i-1]
        offsets.append(offset)
        
        if pre != df['spk'].iloc[i]:
            changes.append(1)
        else:
            changes.append(0)
            
        pre = df['spk'].iloc[i]
            
    return offsets, changes

def get_first_da(da_list, i):
    for da in da_list:
        if da not in ['+f', '+b', 'pf', 'eom']:
            return da
    
    if da=='eom':
        print(names[i], i, da_list)
    return 'pf'

def main(indir, outdir):   
    
    results = {}
    results_vad = {}
    errors = {'name': [], 'duplication': [], 'empty': [], 'id_error': []}
    names = sorted(os.listdir(indir))
    for name in tqdm(names):
        path = os.path.join(indir, name)
        #print(path)
        if '.txt' not in path:
            continue

        df = get_df(path)

        if type(df) != int:

            labels, e1, e2 = select_da_label(df, path)
            df['da'] = labels
            turn_id, e3 = process_turn_id(df)
            df['turn_id'] = turn_id        

            if max(df['turn_id'].tolist())!=1:
                turn_id, _ = modify_turn_id(df)
                df['turn_id'] = turn_id        

            df_agent = df[df['spk']==0]
            df_user = df[df['spk']==1]        

            new_turn_id = check_concat_da(df_agent)
            df_agent['turn_id'] = new_turn_id
            new_turn_id = check_concat_da(df_user)
            df_user['turn_id'] = new_turn_id

            df_vad = pd.concat([df_agent, df_user])
            df_vad = df.sort_values('start')
            vad_list.append(df_vad[['start', 'end', 'spk', 'da', 'turn_id']])

            df_agent = make_turn_df(df_agent, spk_id=0)
            df_user = make_turn_df(df_user, spk_id=1)                

            df = pd.concat([df_agent, df_user])
            df = df.sort_values('start')                

            results[name.replace('.txt', '')] = df
            results_vad[name.replace('.txt', '')] = df_vad
            errors['name'].append(name)
            errors['duplication'].append(e1)
            errors['empty'].append(e2)
            errors['id_error'].append(e3)
            
    names = list(results.keys())
    datas = list(results.values())
    vads = list(results_vad.values())
    
    df_list = []
    da_list = []
    offset_list = []
    cnt = 0
    for i, df in enumerate(tqdm(datas)):
        df['da1'] = df['da'].apply(lambda x: get_first_da(x.split('/'), i))

        cnt += len(df[(df['da1']=='bc') & (df['spk']==0)])
        df = df[df['da1']!='bc']
        offsets, changes = get_offset(df)
        df['offset'] = offsets
        df['spk_change'] = changes
        df = df.iloc[1:]

        nxt_start = df['start'].tolist()[1:]
        nxt_end = df['end'].tolist()[1:]
        nxt_spk = df['spk'].tolist()[1:]

        df = df = df.iloc[:-1]
        df['nxt_start'] = nxt_start
        df['nxt_end'] = nxt_end
        df['nxt_spk'] = nxt_spk            

        # 話者がエージェント, 話者交代した場合に絞る
        #df = df[(df['spk']==0) & (df['spk_change']==1)]
        df_list.append(df)

        da_list += df['da1'].tolist()
        offset_list += df['offset'].tolist()

    print(cnt)
        
    for i, name in enumerate(tqdm(names)):
        df_vad = vads[i]
        df = df_list[i]

        path = os.path.join(outdir, 'csv', name+'.csv')
        path2 = os.path.join(outdir, 'vad', name+'.csv')

        df.to_csv(path, index=False)
        df_vad.to_csv(path2, index=False)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default='/mnt/aoni04/jsakuma/data/ATR_Annotated/annotation_output', help='model type')
    parser.add_argument('--outdir', type=str, default='/mnt/aoni04/jsakuma/data/ATR_Annotated', help='model type')
    args = parser.parse_args()

    main(args.indir, args.outdir)