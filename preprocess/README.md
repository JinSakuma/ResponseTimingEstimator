# Preprocess

* make_annoation_data.py  
  アノテーション結果のtxtファイルから時間, Dialog Act, ターン情報のcsvを作成.  
  引数のindirはGoogleDriveのアノテーション結果のoutput/txt.
  
* make_turn_csv.py  
  make_annoation_data.pyで作成したCSVからターンごとの時間, Dialog Actのcsvを作成. タイミングの範囲の設定などはここで行う (例: -500ms~2000ms).
* make_turn_wav.py  
  make_turn_csv.pyと同様のターンの区切りで音声を切り出す.

