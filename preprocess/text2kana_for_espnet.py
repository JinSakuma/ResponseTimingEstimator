import os
import MeCab
import jaconv
import re
import glob
import codecs
from tqdm import tqdm
from hiragana2id import hiragana2idx, idx2hiragana


tagger = MeCab.Tagger("-Oyomi")
def str2yomi(text):
    yomi_kata = tagger.parse(text).split()[0]
    yomi_hira = jaconv.kata2hira(yomi_kata)
    
    return yomi_hira

phoneme_list = [' ', 'a', 'i', 'u', 'e', 'o', 'a:', 'i:', 'u:', 'e:', 'o:', 'N:',
                'b', 'by', 'k', 'ky', 'g', 'gy', 's', 'sh', 'z', 'zy', 'j',
                't', 'ts', 'ty', 'ch', 'd', 'dy', 'n', 'ny', 'h', 'hy', 'p', 'py', 'm', 'my',
                'y', 'r', 'ry', 'w', 'f', 'q', 'N', ':', '<eou>', '<pad>'
               ]

def phon2idx(phonemes):
    ids = []
    for phon in (phonemes+' <eou>').replace('::', ':').split():
        idx = phoneme_list.index(phon)
        ids.append(idx)
        
    return ids


DATAROOT = '/mnt/aoni04/jsakuma/data/ATR2022/asr'
for split in ['train', 'valid', 'test']:
    path = os.path.join(DATAROOT, 'tmp', split, 'text')
    with open(path) as f:
        lines = f.readlines()
        
    path2 = os.path.join(DATAROOT, 'tmp', split, 'wav.scp')
    with open(path2) as f:
        lines2 = f.readlines()

    path3 = os.path.join(DATAROOT, 'tmp', split, 'utt2spk')
    with open(path3) as f:
        lines3 = f.readlines()

    new_lines = []
    new_lines2 = []
    new_lines3 = []
    for i, line in enumerate(tqdm(lines)):
        try:
            name, text_org = line.split() 
        except:
            if len(line) > 30:
                print(i, 'aaa')
            else:
                print(i, 'bbb')
                print(line)
            continue


        try:        
            text_org = text_org.replace('―', 'ー').replace('－', 'ー').replace('—', 'ー')
            text = re.sub(r"[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]", "", text_org)
            text_removed = re.sub(r"[a-zA-Z]", "", text)    
            code_regex = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]')
            text_clean = code_regex.sub('', text_removed)
            yomi = str2yomi(text_clean)
            hiragana = jaconv.kata2hira(yomi)
            hiragana = ' '.join(idx2hiragana(hiragana2idx(hiragana)))
#             phonemes = jaconv.hiragana2julius(jaconv.kata2hira(yomi))
#             phonemes = phonemes.replace('::', ':')

#             is_removed = False
#             if len(text_org)!=len(text_removed):
#                 is_removed = True

        except:
            continue

#         if not is_removed:
#             if name == '' or phonemes == '':
#                 print('error')
                
        new_lines.append(name+' '+hiragana+'\n')
        new_lines2.append(lines2[i])
        new_lines3.append(lines3[i])


    print('new: {}'.format(len(new_lines)))
    
    new_path = path.replace('tmp', 'data_kana')
    new_path2 = path2.replace('tmp', 'data_kana')
    new_path3 = path3.replace('tmp', 'data_kana')
    for j in tqdm(range(len(new_lines))):
        with codecs.open(new_path,"a","utf-8") as f:
            f.write(new_lines[j])
            
        with codecs.open(new_path2,"a","utf-8") as f:
            f.write(new_lines2[j])
            
        with codecs.open(new_path3,"a","utf-8") as f:
            f.write(new_lines3[j])