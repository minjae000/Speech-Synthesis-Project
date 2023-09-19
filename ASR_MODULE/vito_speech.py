import requests

resp = requests.post(
    'https://openapi.vito.ai/v1/authenticate',
    data={'client_id': 'Y6ytQsvOqDjUkMZPx2L6',
          'client_secret': 'UiFff8NuW3xkDdqyHeI516qAkYjP3Eyx9fpWOkYx'}
)
resp.raise_for_status()
print(resp.json())