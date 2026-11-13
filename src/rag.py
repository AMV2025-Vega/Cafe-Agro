"""RAG sencillo: chunks + TF-IDF (sin descargas). Mejora opcional: embeddings + FAISS."""
from pathlib import Path
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _chunks(text, size=900, overlap=150):
    return [text[i:i + size] for i in range(0, max(len(text), 1), size - overlap) if text[i:i + size].strip()]


class Retriever:
    def __init__(self, folder="corpus"):
        self.items = []
        for p in sorted(Path(folder).glob("**/*")):
            try:
                if p.suffix.lower() == ".pdf":
                    for i, pg in enumerate(PdfReader(p).pages):
                        self.items += [(p.name, i + 1, c) for c in _chunks(pg.extract_text() or "")]
                elif p.suffix.lower() in (".txt", ".md") and p.name.lower() != "readme.md":
                    self.items += [(p.name, 1, c) for c in _chunks(p.read_text(encoding="utf-8"))]
            except Exception as e:
                print(f"[rag] no pude leer {p.name}: {e}")
        if self.items:
            self.vec = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), sublinear_tf=True)
            self.mat = self.vec.fit_transform([c for _, _, c in self.items])

    def search(self, query, k=3):
        if not self.items: return []
        sims = cosine_similarity(self.vec.transform([query]), self.mat)[0]
        top = sims.argsort()[::-1][:k]
        return [{"source": self.items[i][0], "page": self.items[i][1], "score": round(float(sims[i]), 3),
                 "text": self.items[i][2]} for i in top if sims[i] > 0]
