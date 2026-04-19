import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


@dataclass
class Result:
    text: str
    refs: List[str]


class Searcher(ABC):
    def __init__(self, docs: List[str]):
        self.docs = docs
    
    @abstractmethod
    def find(self, q: str, n: int, thresh: float) -> List[str]:
        pass


class KeywordSearch(Searcher):
    def __init__(self, docs: List[str]):
        super().__init__(docs)
        self._vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=8000)
        self._matrix = self._vec.fit_transform(docs)
    
    def find(self, q: str, n: int = 5, thresh: float = 0.08) -> List[str]:
        qv = self._vec.transform([q])
        scores = cosine_similarity(qv, self._matrix).flatten()
        order = np.argsort(scores)[::-1]
        return [self.docs[i] for i in order if scores[i] > thresh][:n]


class SemanticSearch(Searcher):
    def __init__(self, docs: List[str], model: str = "multi-qa-mpnet-base-dot-v1"):
        super().__init__(docs)
        self._encoder = SentenceTransformer(model)
        self._embeddings = self._encoder.encode(docs, normalize_embeddings=True, show_progress_bar=True)
    
    def find(self, q: str, n: int = 5, thresh: float = 0.35) -> List[str]:
        qe = self._encoder.encode([q], normalize_embeddings=True)
        scores = cosine_similarity(qe, self._embeddings).flatten()
        order = np.argsort(scores)[::-1]
        return [self.docs[i] for i in order if scores[i] > thresh][:n]


class LLM:
    def __init__(self, model: str = "gemma-3-27b-it", key: Optional[str] = None):
        self._model = model
        self._key = key or os.getenv("GEMINI_API_KEY")
        if not self._key:
            raise ValueError("Set GEMINI_API_KEY")
        from google import genai
        self._client = genai.Client(api_key=self._key)
    
    def call(self, prompt: str, temp: float = 0.1, max_tok: int = 400) -> str:
        from google.genai import types
        from google.genai.errors import ClientError
        for i in range(3):
            try:
                r = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=temp, max_output_tokens=max_tok)
                )
                time.sleep(2.5)
                return r.text or ""
            except ClientError as e:
                if "429" in str(e) and i < 2:
                    time.sleep(15 * (i + 1))
                else:
                    raise
        return ""


class QA(ABC):
    @abstractmethod
    def ask(self, q: str) -> Result:
        pass


class DirectLLM(QA):
    def __init__(self, llm: Optional[LLM] = None):
        self._llm = llm or LLM()
    
    def ask(self, q: str) -> Result:
        p = f'Answer briefly. If unsure, say "I don\'t know."\n\nQuestion: {q}\n\nAnswer:'
        return Result(text=self._llm.call(p), refs=[])


class RAG(QA):
    def __init__(self, searcher: Searcher, llm: Optional[LLM] = None, k: int = 5, threshold: float = 0.15):
        self._search = searcher
        self._llm = llm or LLM()
        self._k = k
        self._thresh = threshold
    
    def ask(self, q: str) -> Result:
        docs = self._search.find(q, self._k, self._thresh)
        if not docs:
            return Result(text="I don't know.", refs=[])
        
        ctx = "\n---\n".join(docs)
        p = f'''Use ONLY the context below. If unrelated or not found, say "I don't know."

Context:
{ctx}

Question: {q}

Answer:'''
        
        out = self._llm.call(p)
        if "don't know" in out.lower() or "do not know" in out.lower():
            return Result(text=out, refs=[])
        return Result(text=out, refs=docs)


def calc_recall(s: Searcher, qs, k: int = 5, th: float = 0.1) -> float:
    from corpus import get_file, get_page
    hit = 0
    for q in qs:
        for d in s.find(q.text, k, th):
            if get_file(d) == q.file and get_page(d) == q.page:
                hit += 1
                break
    return hit / len(qs) if qs else 0.0


def calc_mrr(s: Searcher, qs, k: int = 5, th: float = 0.1) -> float:
    from corpus import get_file, get_page
    total = 0.0
    for q in qs:
        for rank, d in enumerate(s.find(q.text, k, th), 1):
            if get_file(d) == q.file and get_page(d) == q.page:
                total += 1.0 / rank
                break
    return total / len(qs) if qs else 0.0
