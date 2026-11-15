"""Кыргызча TTS сервери (Meta MMS-TTS, facebook/mms-tts-kir).
Иштетүү:  pip install fastapi uvicorn transformers torch scipy
          uvicorn server:app --host 0.0.0.0 --port 8000
Анан браузерден http://localhost:8000 ачыңыз."""
import io, re
import numpy as np, torch
from scipy.io import wavfile
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import VitsModel, AutoTokenizer, set_seed

tok = AutoTokenizer.from_pretrained("facebook/mms-tts-kir")
model = VitsModel.from_pretrained("facebook/mms-tts-kir").eval()
SR = model.config.sampling_rate

ONES = ["", "бир", "эки", "үч", "төрт", "беш", "алты", "жети", "сегиз", "токуз"]
TENS = ["", "он", "жыйырма", "отуз", "кырк", "элүү", "алтымыш", "жетимиш", "сексен", "токсон"]

def _u1000(n):
    h, r = divmod(n, 100)
    p = []
    if h: p += ["жүз"] if h == 1 else [ONES[h], "жүз"]
    p += [TENS[r // 10], ONES[r % 10]]
    return " ".join(x for x in p if x)

def spell(n):
    if n == 0: return "нөл"
    m, t, r = n // 10**6, (n // 1000) % 1000, n % 1000
    p = []
    if m: p.append(_u1000(m) + " миллион")
    if t: p.append("миң" if t == 1 else _u1000(t) + " миң")
    if r: p.append(_u1000(r))
    return " ".join(p)

def clean(text):
    text = re.sub(r"\d{1,9}", lambda m: " " + spell(int(m.group())) + " ", text.lower())
    text = re.sub(r"[^а-яёңөү\s.,!?\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def chunks(text, limit=220):
    out = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        while len(s) > limit:
            i = s.rfind(" ", 0, limit)
            i = i if i > 0 else limit
            out.append(s[:i]); s = s[i:].strip()
        if s: out.append(s)
    return out

class Req(BaseModel):
    text: str

app = FastAPI()

@app.post("/api/tts")
def tts(req: Req):
    text = clean(req.text)[:2000]
    if not re.search(r"[а-яёңөү]", text):
        raise HTTPException(400, "Текст бош же кыргызча эмес")
    parts = []
    gap = np.zeros(int(SR * 0.25), dtype=np.float32)
    for c in chunks(text):
        set_seed(555)  # бир текст — дайыма бирдей үн
        with torch.no_grad():
            w = model(**tok(c, return_tensors="pt")).waveform[0].numpy()
        parts += [w, gap]
    buf = io.BytesIO()
    wavfile.write(buf, SR, np.concatenate(parts))
    return Response(buf.getvalue(), media_type="audio/wav")

app.mount("/", StaticFiles(directory=".", html=True), name="static")
