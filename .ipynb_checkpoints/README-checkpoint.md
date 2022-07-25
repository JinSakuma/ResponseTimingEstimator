### README

* turn_sort
  * 元となるファイルの作成方法が不明
  * bashで切り出したターン情報をソートするファイル
  * ソートすることで、chの情報を元にターン交代が発生したかどうかを判断できる
  * 出力：turn_info/*_sorted.turn.txt

* stereo2mono
  * userとagentの音声を別々に分離して保存する
  * 出力: wav_mono/*_user.wav, wav_mono/*_user.wav
  
* make_df_and_json
  * start, end, text, offset(発話末から発話タイミングまでの時間[ms])を記録したDataFrameを作成・保存
  * speakerid, wav, dataframeなどのpathをまとめたjsonを作成・保存
  * 出力: dataframe/*.csv, json/*.json