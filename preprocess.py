import librosa
import numpy as np
import os
import pyworld
import pyworld as pw
import glob
from utility import *
import argparse
from datetime import datetime

FEATURE_DIM = 36
SAMPLE_RATE = 16000
FRAMES = 512
FFTSIZE = 1024
SPEAKERS_NUM = len(speakers)
CHUNK_SIZE = 1 # concate CHUNK_SIZE audio clips together
EPSILON = 1e-10
MODEL_NAME = 'starganvc_model'

def load_wavs(dataset: str, sr):
    '''
    data dict contains all audios file path &
    resdict contains all wav files   
    '''
    data = {}
    with os.scandir(dataset) as it:
        for entry in it:
            if entry.is_dir():
                data[entry.name] = []
                # print(entry.name, entry.path)
                with os.scandir(entry.path) as it_f:
                    for onefile in it_f:
                        if onefile.is_file():
                            # print(onefile.path)
                            data[entry.name].append(onefile.path)
    print(f'loaded keys: {data.keys()}')
    #data like {TM1:[xx,xx,xxx,xxx]}
    resdict = {}

    cnt = 0
    for key, value in data.items():
        resdict[key] = {}

        for one_file in value:
            
            filename = one_file.split('/')[-1].split('.')[0] #like 100061
            newkey = f'{filename}'
            wav, _ = librosa.load(one_file, sr=sr, mono=True, dtype=np.float64)
            y,_ = librosa.effects.trim(wav, top_db=15)
            wav = np.append(y[0], y[1:] - 0.97 * y[:-1])

            resdict[key][newkey] = wav
            # resdict[key].append(temp_dict) #like TM1:{100062:[xxxxx], .... }
            print('.', end='')
            cnt += 1

    print(f'\nTotal {cnt} aduio files!')
    return resdict

def chunks(iterable, size):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def wav_to_mcep_file(dataset: str, sr=SAMPLE_RATE, processed_filepath: str = './data/processed'):
    '''convert wavs to mcep feature using image repr'''
    shutil.rmtree(processed_filepath)
    os.makedirs(processed_filepath, exist_ok=True)

    allwavs_cnt = len(glob.glob(f'{dataset}/*/*.wav'))
    print(f'Total {allwavs_cnt} audio files!')

    d = load_wavs(dataset, sr)
    for one_speaker in d.keys():
        values_of_one_speaker = list(d[one_speaker].values())
       
        for index, one_chunk in enumerate (chunks(values_of_one_speaker, CHUNK_SIZE)):
            wav_concated = [] #preserve one batch of wavs
            temp = one_chunk.copy()

            #concate wavs
            for one in temp:
                wav_concated.extend(one)
            wav_concated = np.array(wav_concated)

            #process one batch of wavs 
            f0, ap, sp, coded_sp = cal_mcep(wav_concated, sr=sr, dim=FEATURE_DIM)
            newname = f'{one_speaker}_{index}'
            file_path_z = os.path.join(processed_filepath, newname)
            np.savez(file_path_z, f0=f0, coded_sp=coded_sp)
            print(f'[save]: {file_path_z}')

            #split mcep t0 muliti files  
            for start_idx in range(0, coded_sp.shape[1] - FRAMES + 1, FRAMES):
                one_audio_seg = coded_sp[:, start_idx : start_idx+FRAMES]

                if one_audio_seg.shape[1] == FRAMES:
                    temp_name = f'{newname}_{start_idx}'
                    filePath = os.path.join(processed_filepath, temp_name)

                    np.save(filePath, one_audio_seg)
                    print(f'[save]: {filePath}.npy')
            
        

def world_features(wav, sr, fft_size, dim):
    f0, timeaxis = pyworld.harvest(wav, sr)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, sr,fft_size=fft_size)
    ap = pyworld.d4c(wav, f0, timeaxis, sr, fft_size=fft_size)
    coded_sp = pyworld.code_spectral_envelope(sp, sr, dim)

    return f0, timeaxis, sp, ap, coded_sp

def cal_mcep(wav, sr=SAMPLE_RATE, dim=FEATURE_DIM, fft_size=FFTSIZE):
    '''cal mcep given wav singnal
        the frame_period used only for pad_wav_to_get_fixed_frames
    '''
    f0, timeaxis, sp, ap, coded_sp = world_features(wav, sr, fft_size, dim)
    coded_sp = coded_sp.T # dim x n

    return f0, ap, sp, coded_sp


if __name__ == "__main__":
    start = datetime.now()
    parser = argparse.ArgumentParser(description = 'Convert the wav waveform to mel-cepstral coefficients(MCCs)\
    and calculate the speech statistical characteristics')
    
    input_dir = './data/speakers'
    output_dir = './data/processed'
   
    parser.add_argument('--input_dir', type = str, help = 'the direcotry contains data need to be processed', default = input_dir)
    parser.add_argument('--output_dir', type = str, help = 'the directory stores the processed data', default = output_dir)
    
    argv = parser.parse_args()
    input_dir = argv.input_dir
    output_dir = argv.output_dir

    os.makedirs(output_dir, exist_ok=True)
    
    wav_to_mcep_file(input_dir, SAMPLE_RATE,  processed_filepath=output_dir)

    #input_dir is train dataset. we need to calculate and save the speech\
    # statistical characteristics for each speaker.
    generator = GenerateStatistics(output_dir)
    generator.generate_stats()
    generator.normalize_dataset()
    end = datetime.now()
    print(f"[Runing Time]: {end-start}")
