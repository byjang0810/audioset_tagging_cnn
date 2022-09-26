from pyexpat import model
import h5py
import librosa
import numpy as np
import pandas as pd
import torch
import os
from pathlib import Path
from pytorch.models import Cnn14
import json
import copy




class DataRecorded:
    """오디오 녹음 클래스

    출처 : https://github.com/yinkalario/General-Purpose-Sound-Recognition-Demo/blob/master/audio_detection.py
    출처에서는 실제 마이크를 통해 오디오를 녹음하는 클래스였지만 여기서는 logmel_extract 함수만 사용합니다
    아래 Attributes에 대한 설명도 원문을 가져왔습니다

    Attributes:
        CHANNELS : integer, channels of recordings
        RATE : integer, sample rate
        nfft : integer, nfft for stft
        hsfft : integer, hsfft for fft hop size
        window : string, Defaults to a raised cosine window (‘hann’), which is adequate for most applications in audio signal processing.
        mel_bins : integer, neural network frequency axis
        fmin : integer, lowest frequency (in Hz)
        fmax : integer, highest frequency (in Hz). If None, use fmax = sr / 2.0
    """

    def __init__(
        self,
        CHANNELS=1,
        RATE=32000,
        nfft=1024,
        hsfft=500,
        window="hann",
        mel_bins=64,
        fmin=50,
        fmax=14000,
    ):
        # parameters
        self.CHANNELS = CHANNELS  # channels of recordings
        self.RATE = RATE  # sample rate
        self.nfft = nfft  # nfft for stft
        self.hsfft = hsfft  # hsfft for fft hop size
        self.mel_bins = mel_bins  # neural network frequency axis
        self.window = window
        self.fmin = fmin
        self.fmax = fmax

        self.chunk_ready = None

        self.melW = librosa.filters.mel(
            sr=self.RATE,
            n_fft=self.nfft,
            n_mels=mel_bins,
            fmin=self.fmin,
            fmax=self.fmax,
        )

    def logmel_extract(self):
        """오디오 log mel-spectogram 변환 함수

        오디오를 모델이 처리할 수 있는 feature인 log mel-spectogram으로 변환시켜줍니다
        librosa 라이브러리를 활용합니다

        Args:
        Returns:
            self.chunk_ready를 log mel-spectogram으로 변환 후 반환합니다
        """
        S = (
            np.abs(
                librosa.stft(
                    y=self.chunk_ready,
                    n_fft=self.nfft,
                    hop_length=self.hsfft,
                    center=True,
                    window=self.window,
                    pad_mode="reflect",
                )
            )
            ** 2
        )

        mel_S = np.dot(self.melW, S).T
        log_mel_S = librosa.power_to_db(mel_S, ref=1.0, amin=1e-10, top_db=None)

        return log_mel_S


def load_class_label_indices(class_labels_indices_path):
    """label load 함수

    label이 저장된 csv파일을 읽어 index에 따른 label 사전을 반환합니다
    예시 :
        ix_to_lb : 0 -> Speech
        lb_to_ix : Speech -> 0

    Args:
        class_labels_indices_path: string, class_labels_indices 파일 저장 위치

    Returns:
        labels : label 리스트
        lb_to_ix : label -> index 사전 ex) lb_to_ix["Speech"] = 0
        ix_to_lb : index -> label 사전 ex) ix_to_lb[0] = "Speech"
    """
    df = pd.read_csv(class_labels_indices_path, sep=",")
    labels = df["display_name"].tolist()
    lb_to_ix = {lb: i for i, lb in enumerate(labels)}
    ix_to_lb = {i: lb for i, lb in enumerate(labels)}
    return labels, lb_to_ix, ix_to_lb

class AudioDetection:
    """SED 클래스

    출처 : https://github.com/yinkalario/General-Purpose-Sound-Recognition-Demo/blob/master/audio_detection.py
    출처에서 실시간 녹음 코드를 삭제하고 저장된 wav파일 처리와 후처리를 추가하였습니다
    후처리에는 상위 카테고리로 변환과 사용되지 않는 카테고리 삭제가 포함되어 있습니다

    Attributes:
        seconds : float, 한 번에 sed 인식할 오디오 단위 0.5가 최솟값입니다
        sr : integer, 오디오의 sample rate, pretrained된 모델에 맞춰 32000이 기본값입니다
        max_results : integer, 상위에서 최대 몇 개의 결과를 얻을 것인지 결정합니다.
        cuda : boolean, GPU 사용 여부
        model_name :
            string, 저자의 데모 깃허브에는 pretrained된 모델 2가지가 있는데 어떤 걸 사용할 지 결정합니다
            cnn13, cnn9 가 있으며 이름에서도 볼 수 있듯이 13이 더 큰 모델이고 성능이 더 좋지만 더 느립니다
        pretrained_model_dir : string, pretrained model이 있는 폴더 경로
    """

    def __init__(
        self,
        seconds=0.5,
        sr=16000,
        max_results=2,
        cuda=True,
        model_name="cnn14",
        pretrained_model_dir="/data1/common_datasets/sed/",
    ):
        self.seconds = seconds
        self.sr = sr
        self.cuda = cuda

        num_labels = 527

        if model_name == "cnn13":
            # big, performance
            model_path = os.path.join(
                pretrained_model_dir, "Cnn13_GMP_64x64_520000_iterations_mAP=0.42.pth"
            )
            self.model = Cnn13_GMP_64x64(num_labels)
        elif model_name == "cnn9":
            # small, fast
            model_path = os.path.join(
                pretrained_model_dir, "Cnn9_GMP_64x64_300000_iterations_mAP=0.37.pth"
            )
            self.model = Cnn9_GMP_64x64(num_labels)
        else:
            model_path = os.path.join(pretrained_model_dir, "Cnn14_16k_mAP=0.438.pth")
            self.model = Cnn14(num_labels)

        scalar_fn = os.path.join(pretrained_model_dir, "scalar.h5")
        csv_fname = os.path.join(pretrained_model_dir, "validate_meta.csv")

        # load nn model
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

        self.model.load_state_dict(checkpoint["model"])
        if cuda:
            self.model.cuda()
        self.model.eval()

        self.max_results = max_results
        # load scalar
        with h5py.File(scalar_fn, "r") as hf:
            self.mean = hf["mean"][:]
            self.std = hf["std"][:]

        # load label names
        _, _, self.ix_to_lb = load_class_label_indices(csv_fname)

        pc_path = os.path.join(pretrained_model_dir, "parent_category.json")
        with open(pc_path, "r") as f:
            self.parent_category = json.load(f)

        # self.remain_category = [
        #     "Speech",
        #     "Speech synthesizer",
        #     "Shout",
        #     "Screaming",
        #     "Laughter",
        #     "Crying, sobbing",
        #     "Singing",
        #     "Clapping",
        #     "Music",
        # ]

        self.remain_category = ["Shout, Screaming", "Laughter", "Music"]

        self.result_template = {}
        for cat in self.remain_category:
            self.result_template[cat] = []

        # initialize data format
        self.data = DataRecorded()

    def inference(self, x):
        """sed 인식 함수

        Inference output for single instance from neural network
        미리 학습된 모델을 이용해 sed를 인식합니다

        Args:
            x : 인식할 log mel-spectogram

        Returns:
            predict_idxs : 인식한 label들
            predict_probs : 인식한 label일 확률들
        """
        x = torch.Tensor(x).view(1, x.shape[0], x.shape[1])
        if self.cuda:
            x = x.cuda()

        with torch.no_grad():
            y = self.model(x)

        prob = y.data.cpu().numpy().squeeze(axis=0)
        predict_idxs = prob.argsort()[-self.max_results :][::-1]
        predict_probs = prob[predict_idxs]
        return predict_idxs, predict_probs

    def postprocess(self, res, include_end_time, pps):
        """후처리 함수

        sed한 인식 결과를 받아 구간 합치기와 사용되지 않는 카테고리를 제거하여 반환합니다

        Args:
            res : sed한 인식 결과
            include_end_time: boolean, 결과값에 끝 시간을 포함할 지 결정합니다
            pps :
                integer, post process segments의 줄임말로 sed 인식 결과를 합칠 때 구간에서 몇 개를 무시할 지 결정합니다
                인식 단위가 seconds(기본값 0.5초)이므로 인식 결과가 변동이 심합니다 이를 제거해주기 위하여
                특정 label 사이에 pps 갯수만큼의 다른 label들이 있다면 무시하고 특정 label 하나로 통합합니다
                예를 들면 다음과 같습니다
                Music | Music | Speech | Speech | Music
                이 있다면 pps가 2일 경우 2개를 무시하므로 다음과 같이 하나의 구간으로 바뀝니다.
                Music | Music | Music  | Music  | Music
                pps가 1일 경우 pps보다 사이에 있는 구간 갯수가 더 많으므로 변화가 없습니다
                pps가 0이면 모든 구간을 살리고 연속된 구간 합치기만 진행합니다

        Returns:
            final_res : dict, label별 시작시간이 기록된 dict입니다
            dict {"Speech":[시작시간1, 시작시간2, ...}
            또는 include_end_time이 True라면 다음과 같이 end time 또한 기록합니다
            dict {"Speech":[[시작시간1, 끝시간1], [시작시간2,끝시간2], ...}
            예시 :
                {
                    "Speech" : [0, 10, 20, ...],
                    "Music" : [30, ...],
                    ...
                }
        """
        # 사용되지 않는 카테고리 제거하고 이후 구간 합치기를 위해
        # 가장 prob이 높은 하나만 남깁니다. 이 때 구간을 합치므로 개별 prob는 의미가 없어 제거합니다.
        new_result = []
        for r in res:
            idx = r["idx"]
            sed = r["sed"]
            new_sed = []
            contain_speech = False
            for label in sed:
                if isinstance(label, list):
                    label = label[0]
                if label == "Speech":
                    contain_speech = True
                    break
            if contain_speech:
                continue

            for label in sed:
                if isinstance(label, list):
                    label = label[0]
                # if label == "Speech":
                #     contain_speech = True
                # if contain_speech:  # and label == "Music":
                #     continue
                if label in self.remain_category:
                    new_sed = label
                    break
            if len(new_sed) > 0:
                new_result.append({"idx": idx, "sed": new_sed})
        res = new_result

        after_pp = copy.deepcopy(self.result_template)

        n = 0
        s_time, e_time = 0, 0
        new_seg = True
        while n < len(res):
            r = res[n]
            s, e = r["idx"]
            label = r["sed"]
            if new_seg:
                s_time = s
                e_time = e
                new_seg = False

            an = n + pps + 1
            # 끝
            if (n + 1) == len(res):
                e_time = e
                after_pp[label].append([s_time, e_time] if include_end_time else s_time)
                n += 1
            # 구간 pps만큼 뛰어넘어 합치기
            elif (
                an < len(res)
                and res[an]["idx"][0] == (e + pps * self.seconds)
                and res[an]["sed"] == label
            ):
                n = an
                e_time = res[an]["idx"][1]
            # 연속된 구간 합치기, 중간 빈 구간이 pps 이하임
            elif (
                res[n + 1]["idx"][0] <= (e + pps * self.seconds)
                and res[n + 1]["sed"] == label
            ):
                e_time = res[n + 1]["idx"][1]
                n += 1
            else:
                e_tiem = e
                after_pp[label].append([s_time, e_time] if include_end_time else s_time)
                new_seg = True
                n += 1

        #  minimum_time 미만의 Music 구간은 삭제
        minimum_time = 10
        new_musics = []
        for seg in after_pp["Music"]:
            if seg[1] - seg[0] >= minimum_time:
                new_musics.append(seg)
        after_pp["Music"] = new_musics

        return after_pp

    def __call__(
        self,
        audio,
        save=None,
        save_prob=False,
        include_end_time=True,
        postprocess=True,
        th=0.2,
        get_prob=False,
        pps=1,
    ):
        """sed 실행 함수

        audio를 받아 sed를 실행하고 후처리하여 반환합니다

        Args:
            audio:
                인식할 대상, 음성인식 포맷에 맞는 오디오 경로(string) 또는 numpy array
            save:
                인식 결과 저장 경로, json으로 저장되며 경로에 확장자까지 있어야 합니다
                 ex) result/final.json
                 None이면 저장 안 합니다
            include_end_time: boolean, 결과값에 끝 시간을 포함할 지 결정합니다
            th:
                float,threshold를 의미합니다.
                probability를 기준으로 어떤 값 이상만 결과에 표시할지 결정합니다.
            get_prob: boolean, 결과값에 probability를 포함할 지 결정합니다.
            pps :
                integer, post process segments의 줄임말로 sed 인식 결과를 합칠 때 구간에서 몇 개를 무시할 지 결정합니다
                인식 단위가 seconds(기본값 0.5초)이므로 인식 결과가 변동이 심합니다 이를 제거해주기 위하여
                특정 label 사이에 pps 갯수만큼의 다른 label들이 있다면 무시하고 특정 label 하나로 통합합니다
                예를 들면 다음과 같습니다
                Music | Music | Speech | Speech | Music
                이 있다면 pps가 2일 경우 2개를 무시하므로 다음과 같이 하나의 구간으로 바뀝니다.
                Music | Music | Music  | Music  | Music
                pps가 1일 경우 pps보다 사이에 있는 구간 갯수가 더 많으므로 변화가 없습니다
                pps가 0이면 모든 구간을 살리고 연속된 구간 합치기만 진행합니다

        Returns:
            res : dict, label별 시작시간이 기록된 dict입니다
            dict {"Speech":[시작시간1, 시작시간2, ...}
            또는 include_end_time이 True라면 다음과 같이 end time 또한 기록합니다
            dict {"Speech":[[시작시간1, 끝시간1], [시작시간2,끝시간2], ...}
            예시 :
                {
                    "Speech" : [0, 10, 20, ...],
                    "Music" : [30, ...],
                    ...
                }
        """
        if save_prob:
            get_prob = True
        seconds = self.seconds
        # numpy array
        if isinstance(audio, np.ndarray):
            y = audio
        # wav file
        else:
            y, _ = librosa.load(audio, sr=self.sr)

        res = []
        s = 0
        e = s + seconds

        try:
            while (s * self.sr) < len(y):
                audio = y[int(s * self.sr) : int(e * self.sr)]
                idx = (s, e)

                self.data.chunk_ready = audio
                x = self.data.logmel_extract()
                x = (x - self.mean) / self.std

                predict_idxs, predict_probs = self.inference(x)
                predict_labels = []
                for p_idx in predict_idxs:
                    predict_labels.append(self.ix_to_lb[p_idx])

                # 상위 카테고리로 변환
                labels = []
                for label, prob in zip(predict_labels, predict_probs):
                    prob = float(prob)
                    if prob < th:
                        continue
                    if label in self.parent_category:
                        label = self.parent_category[label]
                    if get_prob:
                        labels.append([label, prob])
                    else:
                        labels.append(label)
                if len(labels) > 0:
                    res.append({"idx": idx, "sed": labels})

                s += seconds
                e += seconds

        except Exception as e:
            print(e)

        if save is not None and save_prob:
            with open(
                save.replace(".json", "_prob.json"), "w", encoding="utf-8"
            ) as json_file:
                json.dump(res, json_file, indent=4, ensure_ascii=False)

        if postprocess:
            res = self.postprocess(res, include_end_time, pps)

        if save is not None:
            with open(save, "w", encoding="utf-8") as json_file:
                json.dump(res, json_file, indent=4, ensure_ascii=False)

        return res

if __name__ == "__main__":
    data = DataRecorded()
    sed = AudioDetection()
    method_list = [func for func in dir(sed) if callable(getattr(sed, func)) and not func.startswith("__")]
    AUDIO = '/data1/common_datasets/stt_sample/3pro_8min.wav'
    sr = sed(AUDIO, include_end_time=True)
    print(sr)
    # sed.inference()
    # sed.postprocess()

    #from modules.apps import AudioDetection

# ad = AudioDetection()
# wav = "/data1/common_datasets/stt/sample/test.wav"
# r = ad(wav,include_end_time=True)
# print(r)
# """
# 리스트 안의 숫자는 [구간 시작 시간,구간 끝 시간] 초 단위
# {
#   'Speech':[[0,10], [30.5,33], ...],
#   'Speech synthesizer':[[10,11], [12,15], ...],
#   'Shout':[],
#   'Screaming':[],
#   'Laughter':[],
#   'Crying, sobbing':[],
#   'Singing':[],
#   'Clapping':[],
#   'Music':[]
# }
    
    #   CHANNELS=1,
        # RATE=32000,
        # nfft=1024,
        # hsfft=500,
        # window="hann",
        # mel_bins=64,
        # fmin=50,
        # fmax=14000,