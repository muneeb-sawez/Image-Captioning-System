import urllib.request
import json

img_path = r'c:\Users\munee\OneDrive\Desktop\DS LAB MINI PROJECT\image_captioning_system\track_a\test_images\2_animal_dog.jpg'

boundary = '----WebKitBoundary12345'
body = bytearray()
body.extend(f'--{boundary}\r\n'.encode('utf-8'))
body.extend(b'Content-Disposition: form-data; name="image"; filename="dog.jpg"\r\n')
body.extend(b'Content-Type: image/jpeg\r\n\r\n')
with open(img_path, 'rb') as f:
    body.extend(f.read())
body.extend(f'\r\n--{boundary}\r\n'.encode('utf-8'))
body.extend(b'Content-Disposition: form-data; name="beams"\r\n\r\n4\r\n')
body.extend(f'--{boundary}--\r\n'.encode('utf-8'))

req = urllib.request.Request('http://127.0.0.1:5000/api/caption', data=body)
req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')

with urllib.request.urlopen(req) as res:
    data = json.loads(res.read())
    print("API RESPONSE:")
    print(json.dumps(data, indent=2))
