from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "knowledge_base" / "banking_support_policies.md"


def load_policy_chunks() -> list[dict]:
    content = POLICY_PATH.read_text(encoding="utf-8")
    sections = content.split("\n## ")[1:]

    chunks = []

    for section in sections:
        lines = section.strip().splitlines()
        title = lines[0].strip()
        guidance = "\n".join(lines[1:]).strip()

        chunks.append(
            {
                "title": title,
                "guidance": guidance,
            }
        )

    return chunks


@lru_cache
def get_policy_index():
    chunks = load_policy_chunks()
    documents = [
        f"{chunk['title']}\n{chunk['guidance']}"
        for chunk in chunks
    ]

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(documents)

    return vectorizer, matrix, chunks


def retrieve_guidance(ticket: str) -> dict:
    vectorizer, policy_matrix, chunks = get_policy_index()

    ticket_vector = vectorizer.transform([ticket])
    scores = cosine_similarity(ticket_vector, policy_matrix)[0]
    best_index = scores.argmax()

    return {
        "policy_title": chunks[best_index]["title"],
        "guidance": chunks[best_index]["guidance"],
        "similarity_score": round(float(scores[best_index]), 3),
    }