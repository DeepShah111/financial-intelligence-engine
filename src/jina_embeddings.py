# Embeddings via the Jina AI API (rate-limited + retry) so a large corpus stays under free-tier limits.

import os
import time
import requests

class JinaEmbeddings:
    # Drop-in replacement for HuggingFaceEmbeddings using Jina's hosted model, paced to avoid 429s.
    def __init__(self, model="jina-embeddings-v3"):
        self.api_key = os.getenv("JINA_API_KEY")
        self.model = model
        self.url = "https://api.jina.ai/v1/embeddings"

    def _embed_once(self, texts):
        # One API call for a small batch of texts.
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": [{"text": t} for t in texts]}
        resp = requests.post(self.url, headers=headers, json=payload, timeout=90)
        if resp.status_code == 429:
            raise requests.HTTPError("429")
        resp.raise_for_status()
        return [item["embedding"] for item in resp.json()["data"]]

    def _embed_with_retry(self, texts, max_retries=6):
        # Retry a batch with exponential backoff when the API is rate-limited (429).
        delay = 5
        for attempt in range(max_retries):
            try:
                return self._embed_once(texts)
            except requests.HTTPError as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    print(f"Jina rate limit hit; waiting {delay}s then retrying...")
                    time.sleep(delay)
                    delay = min(delay * 2, 60)   # back off, cap at 60s
                else:
                    raise

    def embed_documents(self, texts):
        # Embed all documents in small batches with a pause between each to respect the rate limit.
        vectors = []
        batch_size = 8
        total = len(texts)
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            vectors.extend(self._embed_with_retry(batch))
            print(f"Embedded {min(i + batch_size, total)}/{total} chunks")
            time.sleep(1.5)                      # pause between batches to stay under the limit
        return vectors

    def embed_query(self, text):
        # Embed a single query string (with retry).
        return self._embed_with_retry([text])[0]