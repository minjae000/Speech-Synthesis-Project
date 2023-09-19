import os

import sounddevice as sd

# from stt import *
import NMT_MODULE
from TTS_MODULE.tts import *

AUDIO_PATH = '/home/mj/Desktop/mydata/Park_new/'
SCRIPT_PATH = './data/Park_new.txt'
TRANSCRIPT_PATH = './data/Park_new_trans.txt'
SPEAKER_PATH = "/home/mj/Desktop/mydata/04.wav"
SAMPLING_RATE = 22050

import natsort

if __name__ == "__main__":

    # path = '/home/mj/Desktop/mydata/Park_new/'
    # file_list = os.listdir(path)
    # file_list = natsort.natsorted(file_list)
    
    
    
    ## NMT

    
    f = open(SCRIPT_PATH, 'r')
    f2 = open(TRANSCRIPT_PATH, 'r')
    
    script = f.readlines()
    transcript = f2.readlines()

    f.close()
    f2.close()

    print(transcript[::])

    
    ## TTS
    for i, j in enumerate(transcript[::]):

        temp_s = script[i].split('|')
        temp_t = transcript[i].split('|')
        
        print('\n\nScript:', temp_s[1])
        print('Transcript:', temp_t[1], '\n')
        
        # Text_to_Speech()
        Voice_Cloning(temp_t[1])