from pathlib import Path
import re
import sys

import numpy as np
import faiss


BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"


class KnowledgeRetriever:
    def __init__(self):
        print(
            "[RAG] Loading knowledge base...",
            file=sys.stderr,
        )

        self.chunks = []

        self._load_documents()
        self._build_index()

    def _load_documents(self):
        for file_path in KNOWLEDGE_DIR.glob("*.md"):
            text = file_path.read_text(
                encoding="utf-8"
            )

            if not text.strip():
                continue

            paragraphs = [
                paragraph.strip()
                for paragraph in re.split(
                    r"\n\s*\n",
                    text,
                )
                if paragraph.strip()
            ]

            for paragraph in paragraphs:
                self.chunks.append(
                    {
                        "source": file_path.name,
                        "content": paragraph,
                    }
                )

        print(
            f"[RAG] Loaded "
            f"{len(self.chunks)} "
            f"knowledge chunks.",
            file=sys.stderr,
        )

    def _tokenize(self, text):
        return re.findall(
            r"\b[a-zA-Z0-9_]+\b",
            text.lower(),
        )

    def _build_vocabulary(self):
        vocabulary = {}

        for chunk in self.chunks:
            for word in self._tokenize(
                chunk["content"]
            ):
                if word not in vocabulary:
                    vocabulary[word] = len(
                        vocabulary
                    )

        self.vocabulary = vocabulary

    def _text_to_vector(self, text):
        vector = np.zeros(
            len(self.vocabulary),
            dtype=np.float32,
        )

        for word in self._tokenize(text):
            if word in self.vocabulary:
                vector[
                    self.vocabulary[word]
                ] += 1.0

        norm = np.linalg.norm(vector)

        if norm > 0:
            vector /= norm

        return vector

    def _build_index(self):
        if not self.chunks:
            raise RuntimeError(
                "Knowledge base is empty."
            )

        self._build_vocabulary()

        vectors = np.array(
            [
                self._text_to_vector(
                    chunk["content"]
                )
                for chunk in self.chunks
            ],
            dtype=np.float32,
        )

        self.index = faiss.IndexFlatIP(
            vectors.shape[1]
        )

        self.index.add(vectors)

        print(
            f"[RAG] Vocabulary size: "
            f"{len(self.vocabulary)}",
            file=sys.stderr,
        )

        print(
            "[RAG] FAISS index built successfully.",
            file=sys.stderr,
        )

    def search(self, query, top_k=5):
        if not query.strip():
            return []

        query_vector = self._text_to_vector(
            query
        )

        query_vector = np.array(
            [query_vector],
            dtype=np.float32,
        )

        k = min(
            top_k,
            len(self.chunks),
        )

        scores, indices = self.index.search(
            query_vector,
            k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            chunk = self.chunks[index]

            results.append(
                {
                    "source": chunk["source"],
                    "content": chunk["content"],
                    "score": float(score),
                }
            )

        return results


retriever = KnowledgeRetriever()


def search_knowledge(query, top_k=5):
    return retriever.search(
        query=query,
        top_k=top_k,
    )


if __name__ == "__main__":
    print(
        "\n=== AutoPR RAG TEST ===\n"
    )

    query = input(
        "Enter knowledge query: "
    )

    results = search_knowledge(query)

    for i, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"\n--- Result {i} ---"
        )

        print(
            f"Source: {result['source']}"
        )

        print(
            f"Score: "
            f"{result['score']:.4f}"
        )

        print(result["content"])