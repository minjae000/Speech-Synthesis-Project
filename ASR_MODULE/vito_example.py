import json
import requests

config = {
  "use_diarization": True,
  "diarization": {
    "spk_count": 1
  },
  "use_multi_channel": False,
  "use_itn": False,
  "use_disfluency_filter": False,
  "use_profanity_filter": False,
  "use_paragraph_splitter": True,
  "paragraph_splitter": {
    "max": 50
  }
}
resp = requests.post(
    'https://openapi.vito.ai/v1/transcribe',
    headers={'Authorization': 'bearer '+'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE2OTM4MzAxOTQsImlhdCI6MTY5MzgwODU5NCwianRpIjoibzJBWHpLMjBpVGRaTmdUclBsNGgiLCJwbGFuIjoiYmFzaWMiLCJzY29wZSI6InNwZWVjaCIsInN1YiI6Ilk2eXRRc3ZPcURqVWtNWlB4Mkw2In0.dHeBxMrAMjubV24iF2IfpgzDetKL4la1qv4rTWef764'},
    data={'config': json.dumps(config)},
    files={'file': open('/home/mj/Desktop/mydata/04-1.wav', 'rb')}
)
resp.raise_for_status()
print(resp.json())