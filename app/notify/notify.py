import requests
from config import DISCORD_WEBHOOK_URL

# LINE_NOTIFY_TOKEN = 'YOUR_LINE_NOTIFY_TOKEN'  # เปลี่ยนเป็น Token ของคุณ

# def send_line_alert(message):
#     url = 'https://notify-api.line.me/api/notify'
#     headers = {
#         'Authorization': f'Bearer {LINE_NOTIFY_TOKEN}'
#     }
#     data = {'message': message}
#     response = requests.post(url, headers=headers, data=data)
#     return response.status_code == 200

def send_discord_alert(message, image_bytes=None):
    data = {"content": message}
    files = None
    if image_bytes:
        files = {"file": ("unknown.jpg", image_bytes, "image/jpeg")}
    response = requests.post(DISCORD_WEBHOOK_URL, data=data, files=files)
    return response.status_code == 204 or response.status_code == 200