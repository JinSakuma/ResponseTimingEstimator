import wave
import numpy as np
import torch
import os
import glob
from tqdm import tqdm
import sflib.sound.sigproc.spec_image as spec_image
import sflib.speech.feature.autoencoder_pytorch.base as base


class CNNAutoEncoder:
    """
    音声データを与えて画像化するクラス．

    音声を順番に与えることで，
    必要なサンプル数が集まった時点で画像を生成してリストを返す．
    1枚分のデータをまとめて与えて計算させることも可能．
    """

    def __init__(self,
                 framesize=800,
                 frameshift=160,
                 fftsize=1024,
                 image_width=10,
                 image_height=None,
                 image_shift=5,
                 device='cpu'
                ):
        """
        Parameters
        ----------
        framesize : int
            音声のフレームサイズ．デフォルト800（50ms）．
        frameshift : int
            音声のフレームシフト．デフォルト160（10ms）．
        fftsize : int
            FFTのサイズ．デフォルト1024．
        image_width : int
            画像の横幅（何個スペクトルを並べるか）．デフォルト10．
        image_height : int
            画像の縦幅．Noneの場合はfftsize/2になる.
        image_shift : int
            画像のシフト．デフォルト5．                        
        """
        
        if device == 'cpu':
            self.device = torch.device('cuda')
            tr0006_18 = base.load(18, 'csj_0006', 'CSJ0006', device=self.device, map_location='cpu')
        else:
            self.device = torch.device('cpu')
            tr0006_18 = base.load(18, 'csj_0006', 'CSJ0006', device=self.device, map_location='cuda')
        self.ae2 = tr0006_18.autoencoder
        
        self._framesize = framesize
        self._frameshift = frameshift
        self._fftsize = fftsize
        if image_height is None or image_height >= self._fftsize // 2:
            self._image_height = self._fftsize // 2
        else:
            self._image_height = image_height
        self._image_width = image_width
        self._image_shift = image_shift
        
        self.generator = spec_image.SpectrogramImageGenerator(
                     framesize=framesize,
                     frameshift=frameshift,
                     fftsize=fftsize,
                     image_width=image_width,
                     image_height=image_height,
                     image_shift=image_shift
                    )
        self.generator.reset()
        
    def __call__(self, x, streaming=False):
        
        if streaming:
            return self.extract_streaming(x)
        else:
            return self.extract(x)        

    def extract(self, x):
        """
        CNN-AEのボトルネック特徴量を計算する．

        Parameters
        ----------
        x : numpy.ndarray (int16)

        Returns
        -------
        feature : numpy.ndarray (int16) 
        power : numpy.ndarray (int16)
        """
                
        pad = np.zeros(self._framesize, np.int16)
        x = np.concatenate([pad, x, pad])        
        with torch.no_grad():
            #spectrogramの作成
            result = self.generator.input_wave(x)

            power = []
            feature = []
            #中間層出力 (encode)
            for j in range(len(result)):
                image_in = result[j].reshape(1, self._image_height, self._image_width)
                image_in = torch.tensor(image_in).float().to(self.device)
                # 中間層出力
                x, l2 = self.ae2.encode(image_in)
                #print(x.shape)
                power.append(l2[0].detach().cpu().data.numpy())
                feature.append(x[0].detach().cpu().data.numpy())

        power = np.vstack(power)
        feature = np.vstack(feature)
        return feature, power
    
    def extract_streaming(self, x):
        """
        CNN-AEのボトルネック特徴量を計算する．

        Parameters
        ----------
        x : numpy.ndarray (int16)

        Returns
        -------
        feature : numpy.ndarray (int16) 
        power : numpy.ndarray (int16)
        """
        
        power = []
        feature = []
        with torch.no_grad():    
            #spectrogramの作成
            result = self.generator.input_wave(x)                

            image_in = result[0].reshape(1, self._image_height, self._image_width)
            image_in = torch.tensor(image_in).float().to(self.device)
            # 中間層出力
            x, l2 = self.ae2.encode(image_in)
            power.append(l2[0].detach().cpu().data.numpy())
            feature.append(x[0].detach().cpu().data.numpy())

        power = np.vstack(power)
        feature = np.vstack(feature)    
        return feature, power

    def reset_generator(self):
        self.generator.reset()