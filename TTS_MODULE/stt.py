import io
import os
import natsort

from google.cloud import speech


AUDIO_PATH = '/home/mj/Desktop/mydata/Park_new/'


def speech_to_text(file_list):

    client = speech.SpeechClient()

    file_name = os.path.join(AUDIO_PATH + file_list)

    with io.open(file_name, "rb") as audio_file:
        content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)
    
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=22050,
        language_code="ko-KR",
        audio_channel_count=2
    )

    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        script = result.alternatives[0].transcript
        print('\n' + script)
        return script
