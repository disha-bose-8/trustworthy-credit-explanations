# Groq version. In the venv:  pip install groq
# Free key (no card): https://console.groq.com/keys
# PowerShell:  $env:GROQ_API_KEY = "your-key"
#              $env:GROQ_MODEL   = "openai/gpt-oss-120b"
# Only this file depends on the LLM. To use another provider, replace the body of call_llm.
import os, time
from groq import Groq

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
MIN_GAP = float(os.environ.get("GROQ_GAP", "12"))   # seconds between calls (keeps under the token-per-minute cap)

_client = Groq(api_key=os.environ["GROQ_API_KEY"])
_last = 0.0

def call_llm(prompt: str) -> str:
    global _last
    extra = {"reasoning_effort": "low"} if "gpt-oss" in MODEL else {}   # less hidden thinking = fewer tokens
    for attempt in range(4):
        wait = MIN_GAP - (time.time() - _last)
        if wait > 0:
            time.sleep(wait)
        _last = time.time()
        try:
            r = _client.chat.completions.create(
                model=MODEL, temperature=0, max_tokens=1000,   # temperature 0: same input -> same output
                messages=[{"role": "user", "content": prompt}],
                extra_body=extra,
            )
            text = r.choices[0].message.content
            if not text:
                raise ValueError("empty answer from model")
            return text
        except Exception as e:
            print("API error:", str(e)[:200])
            if attempt == 3:
                raise RuntimeError(
                    "Groq kept failing. If this is a daily-limit error, rerun the same command "
                    "later: finished applicants are cached and will be skipped.")
            time.sleep(30 * (attempt + 1))