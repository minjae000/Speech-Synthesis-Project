import os
import natsort

from stt import speech_to_text
from nmt import translate_text
from tts import Voice_Cloning


AUDIO_PATH = '/home/mj/Desktop/mydata/Park_new/'
# SCRIPT_PATH = '/home/mj/Desktop/mydata/Script_new.txt'
SPEAKER_PATH = "/home/mj/Desktop/mydata/04.wav"
SAMPLING_RATE = 22050


if __name__ == "__main__":

    file_list = os.listdir(AUDIO_PATH)
    file_list = natsort.natsorted(file_list)
    
    for i, j in enumerate(file_list):
        transcript = translate_text('en', speech_to_text(j))
        # print(transcript)
        Voice_Cloning(transcript)
    
    # with open(SCRIPT_PATH, 'r') as f:
    #     script = f.readlines()

    
    
