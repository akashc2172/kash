from __future__ import annotations
import os, time
import requests
from typing import Any, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

class CBDClient:
    def __init__(self, api_key: Optional[str]=None, base_url: Optional[str]=None, timeout: int=60, max_retries: int=5):
        self.api_key = api_key or os.getenv("CBD_API_KEY")
        self.base_url = (base_url or os.getenv("CBD_BASE_URL") or "https://api.collegebasketballdata.com").rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        if not self.api_key:
            raise ValueError("Missing CBD_API_KEY (set env var or pass api_key=...)")

    def _headers(self) -> Dict[str,str]:
        # Swagger uses apiKey-style header; adjust if your account uses Bearer.
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }

    def get(self, path: str, params: Optional[Dict[str,Any]]=None) -> Any:
        url = f"{self.base_url}{path}"
        last_err = None
        attempt = 0
        while True:
            attempt += 1
            if attempt > self.max_retries and self.max_retries > 0:
                break
            try:
                r = requests.get(url, headers=self._headers(), params=params or {}, timeout=self.timeout)
                if r.status_code == 429:
                    # rate limit - respect header or default 30s. Exp backoff.
                    wait = float(r.headers.get("Retry-After", 30))
                    # Cap wait at 60s
                    sq_wait = min(60, max(wait, 2 ** min(attempt, 6)))
                    print(f"[429] Rate Limit on {path}. Waiting {sq_wait:.1f}s...")
                    time.sleep(sq_wait)
                    # IMPORTANT: Do not increment attempt for rate limits, or reset it
                    # actually, the simplest is just 'continue' and let attempt grow? 
                    # No, if attempt grows we hit max_retries. 
                    # Let's decrement attempt so it's a "free" retry.
                    attempt -= 1
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                # Network errors? wait a bit
                time.sleep(min(15, 1.0 * attempt))
        raise RuntimeError(f"GET failed {url}: {last_err}")
