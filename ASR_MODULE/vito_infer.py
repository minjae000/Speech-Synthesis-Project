import requests

resp = requests.get(
    'https://openapi.vito.ai/v1/transcribe/'+'NLYQuhbgSwyDmx5JPihzkA',
    headers={'Authorization': 'bearer '+'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE2OTM4Mjk3NTgsImlhdCI6MTY5MzgwODE1OCwianRpIjoiZUptWGRlV1VYVzc0RVgzWjIxemMiLCJwbGFuIjoiYmFzaWMiLCJzY29wZSI6InNwZWVjaCIsInN1YiI6Ilk2eXRRc3ZPcURqVWtNWlB4Mkw2In0.e_v699P4e76BoOrtzUiDTt-P_SJC_gc5r6br753NPeY'},
)
resp.raise_for_status()
print(resp.json())