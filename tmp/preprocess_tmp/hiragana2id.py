import re


kana2idx = {
    '<blank>': 0,
    'あ': 1,
    'い': 2,
    'う': 3,
    'え': 4,
    'お': 5,
    'か': 6,
    'き': 7,
    'く': 8,
    'け': 9,
    'こ': 10,
    'さ': 11,
    'し': 12,
    'す': 13,
    'せ': 14,
    'そ': 15,
    'た': 16,
    'ち': 17,
    'つ': 18,
    'て': 19,
    'と': 20,
    'な': 21,
    'に': 22,
    'ぬ': 23,
    'ね': 24,
    'の': 25,
    'は': 26,
    'ひ': 27,
    'ふ': 28,
    'へ': 29,
    'ほ': 30,
    'ま': 31,
    'み': 32,
    'む': 33,
    'め': 34,
    'も': 35,
    'ら': 36,
    'り': 37,
    'る': 38,
    'れ': 39,
    'ろ': 40,
    'が': 41,
    'ぎ': 42,
    'ぐ': 43,
    'げ': 44,
    'ご': 45,
    'ざ': 46,
    'じ': 47,
    'ず': 48,
    'ぜ': 49,
    'ぞ': 50,
    'だ': 51,
    'ぢ': 52,
    'づ': 53,
    'で': 54,
    'ど': 55,
    'ば': 56,
    'び': 57,
    'ぶ': 58,
    'べ': 59,
    'ぼ': 60,
    'ぱ': 61,
    'ぴ': 62,
    'ぷ': 63,
    'ぺ': 64,
    'ぽ': 65,
    'や': 66,
    'ゆ': 67,
    'よ': 68,
    'わ': 69, 
    'を': 70,
    'ん': 71,
    'っ': 72, 
    'きゃ': 73,
    'きゅ': 74,
    'きょ': 75,
    'しゃ': 76,
    'しゅ': 77,
    'しぇ': 78,
    'しょ': 79,
    'ちゃ': 80,
    'ちゅ': 81,
    'ちぇ': 82,
    'ちょ': 83,
    'つぁ': 84,
    'つぃ': 85,   
    'つぇ': 86,
    'つぉ': 87,       
    'でゃ': 88,
    'でぃ': 89,
    'でゅ': 90,
    'でょ': 91,   
    'にゃ': 92,
    'にゅ': 93,
    'にょ': 94,
    'ひゃ': 95,
    'ひゅ': 96,
    'ひょ': 97,
    'みゃ': 98,
    'みゅ': 99,
    'みょ': 100,
    'りゃ': 101,
    'りゅ': 102,
    'りょ': 103,
    'ぎゃ': 104,
    'ぎゅ': 105,
    'ぎょ': 106,              
    'じゃ': 107,
    'じゅ': 108,
    'じぇ': 109,   
    'じょ': 110,
    'びゃ': 111,
    'びゅ': 112,
    'びょ': 113,
    'ぴゃ': 114,
    'ぴゅ': 115,
    'ぴょ': 116,   
    'うぃ': 117,
    'うぇ': 118,   
    'ふぁ': 119,
    'ふぃ': 120,        
    'ふぇ': 121,
    'ふぉ': 122,
    'ー': 123,
    '<eou>': 124,
    '<pad>': 125,
}

idx2kana = dict(zip(list(kana2idx.values()), list(kana2idx.keys())))
    
    
JULIUS_LONG_VOWEL = tuple(
    (
        (re.compile('( a){2,}'), ' 1 123'),
        (re.compile('( i){2,}'), ' 2 123'),
        (re.compile('( u){2,}'), ' 3 123'),
        (re.compile('( e){2,}'), ' 4 123'),
        (re.compile('( o){2,}'), ' 5 123')
    )
)

def idx2hiragana(ids):    
    text = [idx2kana[i] for i in ids]    
    return text

def hiragana2idx(text):
    """Convert Hiragana to Julius's phoneme format.
    Parameters
    ----------
    text : str
        Hiragana string.
    Return
    ------
    str
        Alphabet string.
    Examples
    --------
    >>> print(jaconv.hiragana2julius('てんきすごくいいいいいい'))
    t e N k i s u g o k u i:
    """

    # 3文字以上からなる変換規則
    text = text.replace('う゛ぁ', 'ば')
    text = text.replace('う゛ぃ', 'び')
    text = text.replace('う゛ぇ', 'べ')
    text = text.replace('う゛ぉ', 'ぼ')
    text = text.replace('う゛ゅ', 'びゅ')

    # 2文字からなる変換規則
    text = text.replace('ぅ゛', 'ぶ')

    text = text.replace('あぁ', 'ああ')
    text = text.replace('いぃ', 'いい')
    text = text.replace('いぇ', 'いえ')
    text = text.replace('いゃ', 'や')
    text = text.replace('うぅ', 'う:')
    text = text.replace('えぇ', 'ええ')
    text = text.replace('おぉ', 'おお')
    text = text.replace('かぁ', 'か:')
    text = text.replace('きぃ', 'き:')
    text = text.replace('くぅ', 'く:')
    text = text.replace('くゃ', 'きゃ')
    text = text.replace('くゅ', 'きゅ')
    text = text.replace('くょ', 'きょ')
    text = text.replace('けぇ', 'け:')
    text = text.replace('こぉ', 'こ:')
    text = text.replace('がぁ', 'が:')
    text = text.replace('ぎぃ', 'ぎ:')
    text = text.replace('ぐぅ', 'ぐ:')
    text = text.replace('ぐゃ', 'ぎゃ')
    text = text.replace('ぐゅ', 'ぎゅ')
    text = text.replace('ぐょ', 'ぎょ')
    text = text.replace('げぇ', 'げ:')
    text = text.replace('ごぉ', 'ご:')
    text = text.replace('さぁ', 'さ:')
    text = text.replace('しぃ', 'し:')
    text = text.replace('すぅ', 'す:')
    text = text.replace('すゃ', 'しゃ')
    text = text.replace('すゅ', 'しゅ')
    text = text.replace('すょ', 'しょ')
    text = text.replace('せぇ', 'せ:')
    text = text.replace('そぉ', 'そ:')
    text = text.replace('ざぁ', 'ざ:')
    text = text.replace('じぃ', 'じ:')
    text = text.replace('ずぅ', 'ず:')
    text = text.replace('ずゃ', 'じゃ')
    text = text.replace('ずゅ', 'じゅ')
    text = text.replace('ずょ', 'じょ')
    text = text.replace('ぜぇ', 'ぜ:')
    text = text.replace('ぞぉ', 'ぞ:')
    text = text.replace('たぁ', 'た:')
    text = text.replace('ちぃ', 'ち:')    
    text = text.replace('てぇ', 'て:')
    text = text.replace('とぉ', 'と:')
    text = text.replace('だぁ', 'だ:')
    text = text.replace('ぢぃ', 'じ:')
    text = text.replace('づぅ', 'づ:')
    text = text.replace('づゃ', 'じゃ')
    text = text.replace('づゅ', 'じゅ')
    text = text.replace('づょ', 'じょ')
    text = text.replace('でぇ', 'で:')
    text = text.replace('どぉ', 'ど:')
    text = text.replace('なぁ', 'な:')
    text = text.replace('にぃ', 'に:')
    text = text.replace('ぬぅ', 'ぬ:')
    text = text.replace('ぬゃ', 'にゃ')
    text = text.replace('ぬゅ', 'にゅ')
    text = text.replace('ぬょ', 'にょ')
    text = text.replace('ねぇ', 'ね:')
    text = text.replace('のぉ', 'の:')
    text = text.replace('はぁ', 'は:')
    text = text.replace('ひぃ', 'ひ:')
    text = text.replace('ふぅ', 'ふ:')
    text = text.replace('ふゃ', 'ひゃ')
    text = text.replace('ふゅ', 'ひゅ')
    text = text.replace('ふょ', 'ひょ')
    text = text.replace('へぇ', 'へ:')
    text = text.replace('ほぉ', 'ほ:')
    text = text.replace('ばぁ', 'ば:')
    text = text.replace('びぃ', 'び:')
    text = text.replace('ぶぅ', 'ぶ:')
    text = text.replace('ふゃ', 'ひゃ')
    text = text.replace('ぶゅ', 'びゅ')
    text = text.replace('ふょ', 'ひょ')
    text = text.replace('べぇ', 'べ:')
    text = text.replace('ぼぉ', 'ぼ:')
    text = text.replace('ぱぁ', 'ぱ:')
    text = text.replace('ぴぃ', 'ぴ:')
    text = text.replace('ぷぅ', 'ぷ:')
    text = text.replace('ぷゃ', 'ぴゃ')
    text = text.replace('ぷゅ', 'ぴゅ')
    text = text.replace('ぷょ', 'ぴょ')
    text = text.replace('ぺぇ', 'ぺ:')
    text = text.replace('ぽぉ', 'ぽ:')
    text = text.replace('まぁ', 'ま:')
    text = text.replace('みぃ', 'み:')
    text = text.replace('むぅ', 'む:')
    text = text.replace('むゃ', 'みゃ')
    text = text.replace('むゅ', 'みゅ')
    text = text.replace('むょ', 'みょ')
    text = text.replace('めぇ', 'め:')
    text = text.replace('もぉ', 'も:')
    text = text.replace('やぁ', 'や:')
    text = text.replace('ゆぅ', 'ゆ:')
    text = text.replace('ゆゃ', 'や:')
    text = text.replace('ゆゅ', 'ゆ:')
    text = text.replace('ゆょ', 'よ:')
    text = text.replace('よぉ', 'よ:')
    text = text.replace('らぁ', 'ら:')
    text = text.replace('りぃ', 'り:')
    text = text.replace('るぅ', 'る:')
    text = text.replace('るゃ', 'りゃ')
    text = text.replace('るゅ', 'りゅ')
    text = text.replace('るょ', 'りょ')
    text = text.replace('れぇ', 'れ:')
    text = text.replace('ろぉ', 'ろ:')
    text = text.replace('わぁ', 'わ:')
    text = text.replace('をぉ', 'お:')
    text = text.replace('う゛', 'ぶ')    
    text = text.replace('でぇ', 'でー')  
    text = text.replace('どゃ', 'でゃ')
    text = text.replace('どゅ', 'でゅ')
    text = text.replace('どょ', 'でょ')
    text = text.replace('てぃ', 'ち')
    text = text.replace('てぇ', 'て:')
    text = text.replace('てゃ', 'ちゃ')
    text = text.replace('てゅ', 'ちゅ')
    text = text.replace('てょ', 'ちょ')
    text = text.replace('すぃ', 'し')
    text = text.replace('ずぁ', 'ずあ')
    text = text.replace('ずぃ', 'じ')
    text = text.replace('ずぅ', 'ず')
    text = text.replace('ずゃ', 'じゃ')
    text = text.replace('ずゅ', 'じゅ')
    text = text.replace('ずょ', 'じょ')
    text = text.replace('ずぇ', 'ぜ')
    text = text.replace('ずぉ', 'ぞ')
    text = text.replace('つゃ', ' ちゃ')
    text = text.replace('つゅ', ' ちゅ')
    text = text.replace('つょ', ' ちょ')
    text = text.replace('とぅ', 'つ')
    text = text.replace('とゃ', ' ちゃ')
    text = text.replace('とゅ', ' ちゅ')
    text = text.replace('とょ', ' ちょ')
    text = text.replace('どぁ', 'どあ')
    text = text.replace('どぅ', 'づ')
    text = text.replace('どぉ', 'ど:')
    text = text.replace('つぅ', 'つ:') 
    text = text.replace('ぢゃ', 'じゃ')
    text = text.replace('ぢゅ', 'じゅ')
    text = text.replace('ぢぇ', 'じぇ')
    text = text.replace('ぢょ', 'じょ')
    text = text.replace('うぁ', ' うあ')
    text = text.replace('うぉ', ' を')
    text = text.replace('ふぅ', 'ふ')
    text = text.replace('ふゃ', ' ひゃ')
    text = text.replace('ふゅ', ' ひゅ')
    text = text.replace('ふょ', ' ひょ')
    
    text = text.replace('きゃ', ' 73')
    text = text.replace('きゅ', ' 74')
    text = text.replace('きょ', ' 75')
    text = text.replace('しゃ', ' 76')
    text = text.replace('しゅ', ' 77')
    text = text.replace('しぇ', ' 78')
    text = text.replace('しょ', ' 79')
    text = text.replace('ちゃ', ' 80')
    text = text.replace('ちゅ', ' 81')
    text = text.replace('ちぇ', ' 82')
    text = text.replace('ちょ', ' 83')
    text = text.replace('つぁ', ' 84')
    text = text.replace('つぃ', ' 85')    
    text = text.replace('つぇ', ' 86')
    text = text.replace('つぉ', ' 87')        
    text = text.replace('でゃ', ' 88')
    text = text.replace('でぃ', ' 89')
    text = text.replace('でゅ', ' 90')
    text = text.replace('でょ', ' 91')    
    text = text.replace('にゃ', ' 92')
    text = text.replace('にゅ', ' 93')
    text = text.replace('にょ', ' 94')
    text = text.replace('ひゃ', ' 95')
    text = text.replace('ひゅ', ' 96')
    text = text.replace('ひょ', ' 97')
    text = text.replace('みゃ', ' 98')
    text = text.replace('みゅ', ' 99')
    text = text.replace('みょ', ' 100')
    text = text.replace('りゃ', ' 101')
    text = text.replace('りゅ', ' 102')
    text = text.replace('りょ', ' 103')
    text = text.replace('ぎゃ', ' 104')
    text = text.replace('ぎゅ', ' 105')
    text = text.replace('ぎょ', ' 106')               
    text = text.replace('じゃ', ' 107')
    text = text.replace('じゅ', ' 108')
    text = text.replace('じぇ', ' 109')    
    text = text.replace('じょ', ' 110')
    text = text.replace('びゃ', ' 111')
    text = text.replace('びゅ', ' 112')
    text = text.replace('びょ', ' 113')
    text = text.replace('ぴゃ', ' 114')
    text = text.replace('ぴゅ', ' 115')
    text = text.replace('ぴょ', ' 116')    
    text = text.replace('うぃ', ' 117')
    text = text.replace('うぇ', ' 118')    
    text = text.replace('ふぁ', ' 119')
    text = text.replace('ふぃ', ' 120')        
    text = text.replace('ふぇ', ' 121')
    text = text.replace('ふぉ', ' 122')

    # 1音からなる変換規則
    text = text.replace('ゐ', ' い')
    text = text.replace('ゑ', ' え')
    
    text = text.replace('あ', ' 1')
    text = text.replace('い', ' 2')
    text = text.replace('う', ' 3')
    text = text.replace('え', ' 4')
    text = text.replace('お', ' 5')
    text = text.replace('か', ' 6')
    text = text.replace('き', ' 7')
    text = text.replace('く', ' 8')
    text = text.replace('け', ' 9')
    text = text.replace('こ', ' 10')
    text = text.replace('さ', ' 11')
    text = text.replace('し', ' 12')
    text = text.replace('す', ' 13')
    text = text.replace('せ', ' 14')
    text = text.replace('そ', ' 15')
    text = text.replace('た', ' 16')
    text = text.replace('ち', ' 17')
    text = text.replace('つ', ' 18')
    text = text.replace('て', ' 19')
    text = text.replace('と', ' 20')
    text = text.replace('な', ' 21')
    text = text.replace('に', ' 22')
    text = text.replace('ぬ', ' 23')
    text = text.replace('ね', ' 24')
    text = text.replace('の', ' 25')
    text = text.replace('は', ' 26')
    text = text.replace('ひ', ' 27')
    text = text.replace('ふ', ' 28')
    text = text.replace('へ', ' 29')
    text = text.replace('ほ', ' 30')
    text = text.replace('ま', ' 31')
    text = text.replace('み', ' 32')
    text = text.replace('む', ' 33')
    text = text.replace('め', ' 34')
    text = text.replace('も', ' 35')
    text = text.replace('ら', ' 36')
    text = text.replace('り', ' 37')
    text = text.replace('る', ' 38')
    text = text.replace('れ', ' 39')
    text = text.replace('ろ', ' 40')
    text = text.replace('が', ' 41')
    text = text.replace('ぎ', ' 42')
    text = text.replace('ぐ', ' 43')
    text = text.replace('げ', ' 44')
    text = text.replace('ご', ' 45')
    text = text.replace('ざ', ' 46')
    text = text.replace('じ', ' 47')
    text = text.replace('ず', ' 48')
    text = text.replace('ぜ', ' 49')
    text = text.replace('ぞ', ' 50')
    text = text.replace('だ', ' 51')
    text = text.replace('ぢ', ' 52')
    text = text.replace('づ', ' 53')
    text = text.replace('で', ' 54')
    text = text.replace('ど', ' 55')
    text = text.replace('ば', ' 56')
    text = text.replace('び', ' 57')
    text = text.replace('ぶ', ' 58')
    text = text.replace('べ', ' 59')
    text = text.replace('ぼ', ' 60')
    text = text.replace('ぱ', ' 61')
    text = text.replace('ぴ', ' 62')
    text = text.replace('ぷ', ' 63')
    text = text.replace('ぺ', ' 64')
    text = text.replace('ぽ', ' 65')
    text = text.replace('や', ' 66')
    text = text.replace('ゆ', ' 67')
    text = text.replace('よ', ' 68')
    text = text.replace('わ', ' 69')  
    text = text.replace('を', ' 70')
    text = text.replace('ん', ' 71')
    text = text.replace('っ', ' 72')
    # ここまでに処理されてない ぁぃぅぇぉ はそのまま大文字扱い
    text = text.replace('ぁ', ' 1')
    text = text.replace('ぃ', ' 2')
    text = text.replace('ぅ', ' 3')
    text = text.replace('ぇ', ' 4')
    text = text.replace('ぉ', ' 5')
    text = text.replace('ゎ', ' 69')
    text = text.replace('ぉ', ' 5')

    # 長音の処理
    for (pattern, replace_str) in JULIUS_LONG_VOWEL:
        text = pattern.sub(replace_str, text)
#     text = text.replace('o u', 'o:')  # おう -> おーの音便
    text = text.replace('ー', ' 123')
    text = text.replace('〜', ' 123')
    text = text.replace('−', ' 123')
    text = text.replace('-', ' 123')
    text = text.replace('::', ' 123')
    text = text.replace(':::', ' 123')
    
    text = text.replace('<eou>', ' 124')


    #その他特別な処理
    text = text.replace('を', ' 70')

    text = text.strip()

    text = text.replace(':+', '123')
    
    ids = [int(i) for i in text.split()]
    
    return ids
