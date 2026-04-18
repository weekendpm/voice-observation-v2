import httpx
import base64
import os
from dotenv import load_dotenv

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

calls = [
    ("call_01_code_mixed", "Mera account reset karo please"),
    ("call_02_clear_english", "I need to reset my account password"),
    ("call_03_noisy", "asdkfj asldfkj reset kar do yaar please help"),
]

os.makedirs("api/demo_calls", exist_ok=True)

for name, text in calls:
    resp = httpx.post(
        "https://api.sarvam.ai/text-to-speech",
        headers={"api-subscription-key": SARVAM_API_KEY},
        json={"inputs": [text], "target_language_code": "hi-IN", "speaker": "aditya", "model": "bulbul:v3"},
        timeout=30,
    )
    resp.raise_for_status()
    audio_b64 = resp.json()["audios"][0]
    audio_bytes = base64.b64decode(audio_b64)
    with open(f"api/demo_calls/{name}.wav", "wb") as f:
        f.write(audio_bytes)
    print(f"Generated {name}.wav")
