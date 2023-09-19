import torch

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from TTS.api import TTS
from TTS.utils import audio

import sounddevice as sd


OUTPUT_PATH = "/home/mj/Desktop/output_project/"
SPEAKER_PATH = "/home/mj/Desktop/mydata/04_(enhanced).wav"


model = TTS.list_models()[0]
tts = TTS(model, progress_bar=False, gpu=True)


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

hps = utils.get_hparams_from_file("./configs/vctk_base.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint("./pretrained/pretrained_vctk.pth", net_g, None)


# def Text_to_Speech():

#     stn_tst = get_text(temp_t[1], hps)

#     with torch.no_grad():
#         x_tst = stn_tst.cuda().unsqueeze(0)
#         x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
#         sid = torch.LongTensor([28]).cuda()
#         audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

#     # ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))
#     # write(os.path.join('./output/' + file_list[i]), hps.data.sampling_rate, audio)
#     start = time.time()
#     sd.play(audio, hps.data.sampling_rate, blocking=True)
#     end = time.time()
#     print(f"\nSpeech Len : {end - start:.2f} sec")

#     time.sleep(1)
#     # sd.stop()
    

def Voice_Cloning(input_text):
    
    wav = tts.tts(input_text, speaker_wav=SPEAKER_PATH, language="en")
    
    # start = time.time()
    sd.play(wav, 16000, blocking=True)
    # end = time.time()
    # print(f"\nSpeech Len : {end - start:.2f} sec")

    # print(get_text(temp_t[1], hps))
    return wav