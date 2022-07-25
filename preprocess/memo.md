### README

* make_data_for_julius.py
  * 書き起こしのひらがな化→txt形式で保存
  * wavの保存
  * 同一ファイルに保存→julius
  
* run_julius.py
  * julius/segmentation-kit直下において使う
  * make_data_for_julius.pyでwavとtxtを保存したdirを引数に渡すとアライメントを取って.labファイルと.logファイルを作成
  
* julius2text.py
  * juliusのアライメント結果を元にagentの書き起こしをVAD結果(ターン単位)に対応付ける
  * juliusがうまく行ってない場合(一部しかアライメントされてないなど)にうまくいかないため改良or手直しが必要

* make_data_for_espnet.py
  * agentの音声をespnetのデータ形式にする(Text, wav.scp, utt2spkなど)
  * user音声Verはどこかにあるはず...
  
* wav2text.py
  * espnetで学習したモデルでdecodeしてテキストを保存する
  
* make_df_and_json.py
  * start, end, text, offset(発話末から発話タイミングまでの時間[ms])などを記録したDataFrameを作成・保存
  * speakerid, wav, dataframeなどのpathをまとめたjsonを作成・保存
  * 出力: dataframe/*.csv, json/*.json
  
* save_turn_wav.py
  * make_df_and_json.pyで保存したターン情報を元にターン単位のwavを切り抜く
  
* add_text_to_df.py
  * agentのjulius結果、userのasr結果のテキストをmake_df_and_json.pyで作ったdfに追加して別ファイルとして保存