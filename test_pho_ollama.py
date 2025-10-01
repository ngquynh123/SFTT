import requests

url = "http://localhost:11434/api/generate"

payload = {
    "model": "mrjacktung/phogpt-4b-chat-gguf",
    "prompt": "Phanh tay trên ô tô có chức năng chính là gì?"
}

resp = requests.post(url, json=payload, stream=True)

for line in resp.iter_lines():
    if line:
        print(line.decode("utf-8"))
