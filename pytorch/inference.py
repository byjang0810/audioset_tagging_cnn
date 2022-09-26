import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch
from argparse import Namespace
from utilities import create_folder, get_filename
from models import *
from pytorch_utils import move_data_to_device
import config
import copy
import json

def audio_tagging(args):
    """Inference audio tagging result of an audio clip.
    """

    # Arugments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    
    classes_num = config.classes_num
    labels = config.labels

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    if 'cuda' in str(device):
        model.to(device)
        print('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')
    
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
    """(classes_num,)"""

    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    for k in range(10):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], 
            clipwise_output[sorted_indexes[k]]))

    # Print embedding
    if 'embedding' in batch_output_dict.keys():
        embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]
        print('embedding: {}'.format(embedding.shape))

    return clipwise_output, labels


def sound_event_detection(args):
    """Inference sound event detection result of an audio clip.
    """

    # Arugments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    classes_num = config.classes_num
    labels = config.labels
    frames_per_second = sample_rate // hop_size

    # Paths
    fig_path = os.path.join('results', '{}.png'.format(get_filename(audio_path)))
    create_folder(os.path.dirname(fig_path))

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)
    
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)
    print(batch_output_dict.keys())
    framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
    """(time_steps, classes_num)"""

    print('Sound event detection result (time_steps x classes_num): {}'.format(
        framewise_output.shape))

    sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]

    top_k = 10  # Show top results
    top_result_mat = framewise_output[:, sorted_indexes[0 : top_k]]    
    """(time_steps, top_k)"""

    # Plot result    
    stft = librosa.core.stft(y=waveform[0].data.cpu().numpy(), n_fft=window_size, 
        hop_length=hop_size, window='hann', center=True)
    frames_num = stft.shape[-1]
    print(f"frames_num: {frames_num}")

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
    axs[0].matshow(np.log(np.abs(stft)), origin='lower', aspect='auto', cmap='jet')
    axs[0].set_ylabel('Frequency bins')
    axs[0].set_title('Log spectrogram')
    axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
    axs[1].xaxis.set_ticks(np.arange(0, frames_num, frames_per_second))
    axs[1].xaxis.set_ticklabels(np.arange(0, frames_num / frames_per_second))
    axs[1].yaxis.set_ticks(np.arange(0, top_k))
    axs[1].yaxis.set_ticklabels(np.array(labels)[sorted_indexes[0 : top_k]])
    axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
    axs[1].set_xlabel('Seconds')
    axs[1].xaxis.set_ticks_position('bottom')

    plt.tight_layout()
    plt.savefig(fig_path)
    print('Save sound event detection visualization to {}'.format(fig_path))

    return framewise_output, labels


def postprocess(res, include_end_time, pps, seconds):
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
        # remain_category = ["Speech", "Music"]

        # new_result = []
        # for r in res:
        #     idx = r["idx"]
        #     sed = r["sed"]
        #     new_sed = []
        #     contain_speech = False
        #     for label in sed:
        #         if isinstance(label, list):
        #             label = label[0]
        #         if label == "Speech":
        #             contain_speech = True
        #             break
        #     if contain_speech:
        #         continue

        #     for label in sed:
        #         if isinstance(label, list):
        #             label = label[0]
           
        #         if label in remain_category:
        #             new_sed = label
        #             break
        #     if len(new_sed) > 0:
        #         new_result.append({"idx": idx, "sed": new_sed})
        # res = new_result


        # result_template = {"Music":[], "Speech":[], "Silence": [], "Giggle": []}
        result_template = {s:[] for s in total_labels}
        after_pp = copy.deepcopy(result_template)
      
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
                and res[an]["idx"][0] == (e + pps * seconds)
                and res[an]["sed"] == label
            ):
                n = an
                e_time = res[an]["idx"][1]
            # 연속된 구간 합치기, 중간 빈 구간이 pps 이하임
            elif (
                res[n + 1]["idx"][0] <= (e + pps * seconds)
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

def main():
    #args = Namespace(audio_path='/home/boyoon/audio/dataset/959_cheongchun_conv.wav', checkpoint_path="Cnn14_DecisionLevelMax_mAP=0.385.pth", cuda=True, sample_rate=16000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, mode='sound_event_detection', model_type="Cnn14_DecisionLevelMax")
    args = Namespace(audio_path='/home/boyoon/audio/speaker-diarization/3pro.wav', checkpoint_path="Cnn14_DecisionLevelMax_mAP=0.385.pth", cuda=True, sample_rate=16000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, mode='sound_event_detection', model_type="Cnn14_DecisionLevelMax")   
    framewise_output, labels = sound_event_detection(args)
    max_output = np.argmax(framewise_output, axis=1)
    
    total_duration = librosa.get_duration(filename=args.audio_path)
    time_steps = framewise_output.shape[0] # num of frames
    frame_sec = total_duration / time_steps
    print(max_output)
    
    # print(f"duration: {total_duration}")
    # print(time_steps)
   
    total_result = []
    temp = dict()
    temp['idx'] = [0,0]

    for i in range(1, time_steps): # 1 ~ 47438
        if max_output[i] == max_output[i-1]:
            temp['idx'][1] += frame_sec
        else:
            label = labels[max_output[i-1]]
            temp['sed'] = label
            total_result.append(temp)
            temp = dict()
            temp['idx'] = [0,0]
            temp['idx'][0] = i * frame_sec
            temp['idx'][1] = (i+1) * frame_sec
            
    label = labels[max_output[-1]]
    temp['sed'] = label
    total_result.append(temp)
    return total_result, max_output
 


if __name__ == '__main__':
    main()
    