import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
import fitz


def load_pdf(path: str) -> List[str]:
    doc = fitz.open(path)
    name = Path(path).name
    result = []
    for i, p in enumerate(doc, 1):
        content = p.get_text().strip()
        if content:
            result.append(f"[Source: {name} | Page {i}]\n\n{content}")
    doc.close()
    return result


def load_documents(directory: str) -> List[str]:
    files = sorted(Path(directory).glob("*.pdf"))
    all_chunks = []
    for f in files:
        chunks = load_pdf(str(f))
        all_chunks.extend(chunks)
    return all_chunks


@dataclass
class Question:
    text: str
    answer: str
    file: str
    page: int


QUESTIONS = [
    Question(
        "What philosophical puzzle asks whether holiness is loved because it's holy or holy because it's loved?",
        "Whether pious things are loved by gods because they're pious, or pious because loved by gods",
        "095d54aa351c9a2e1a79d2e8a1ce0201_MIT24_231F09_lec02.pdf", 1
    ),
    Question(
        "What ethical theory claims that God's will determines what is morally good, and what problems does it face?",
        "Divine Command Theory faces problems of arbitrariness, triviality, and abhorrent commands",
        "095d54aa351c9a2e1a79d2e8a1ce0201_MIT24_231F09_lec02.pdf", 2
    ),
    Question(
        "How did the British philosopher argue that goodness cannot be reduced to natural properties?",
        "Any identity between good and natural properties invites an open question that shows they can't be identical",
        "4d3b71ffcc7a256a2dcee44f97277875_MIT24_231F09_lec03.pdf", 1
    ),
    Question(
        "What is the difference between reasons that apply to everyone versus reasons tied to individual perspective?",
        "Agent-neutral reasons don't include essential reference to the person who has them, agent-relative reasons do",
        "2aa5e59ac0dd208cb8d8d58c872aa228_MIT24_231F09_lec18.pdf", 1
    ),
    Question(
        "According to the course, what types of personal motivations limit what morality can demand from us?",
        "Reasons of autonomy, deontological reasons, and reasons of obligation",
        "2aa5e59ac0dd208cb8d8d58c872aa228_MIT24_231F09_lec18.pdf", 1
    ),
    Question(
        "Why might following the principle of maximizing good actually destroy the practice of keeping promises?",
        "Acting on AU would undermine trust needed for truth-telling and promise-keeping to be valuable",
        "5a1e538bccd9add8d5abad4c21fc4a1c_MIT24_231F09_lec16.pdf", 1
    ),
    Question(
        "Why can't we know if our actions are truly right when judging by their outcomes?",
        "We're clueless about actual consequences due to massive causal ramification",
        "53a0477659ae023edf3de9c60975f622_MIT24_231F09_lec15.pdf", 1
    ),
]

UNANSWERABLE = [
    "What percentage of students passed the Ethics 24.231 final exam?",
    "How many office hours per week did the professor hold for this course?",
]

OFFTOPIC = ["What is the syntax for a for-loop in Python?"]


def parse_metadata(chunk: str) -> Tuple[Optional[str], Optional[int]]:
    m = re.search(r"\[Source:\s*([^\|]+)\s*\|\s*Page\s*(\d+)\]", chunk)
    return (m.group(1).strip(), int(m.group(2))) if m else (None, None)


def get_file(chunk: str) -> Optional[str]:
    f, _ = parse_metadata(chunk)
    return f


def get_page(chunk: str) -> Optional[int]:
    _, p = parse_metadata(chunk)
    return p


def get_all_files(docs: List[str]) -> Set[str]:
    files = set()
    for d in docs:
        f = get_file(d)
        if f:
            files.add(f)
    return files


def clean(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return re.sub(r"[^\w\s]", "", t)


def is_uncertain(text: str) -> bool:
    c = clean(text)
    markers = {"i dont know", "i do not know", "dont know", "unknown", "not sure", "cannot answer"}
    return any(m in c for m in markers) or c == ""


def score_answered(resp, q: Question, valid: Set[str]) -> float:
    if not resp.refs:
        return 0.0
    for r in resp.refs:
        if get_file(r) == q.file and get_page(r) == q.page:
            return 1.0
    for r in resp.refs:
        if get_file(r) in valid:
            return 0.2
    return 0.0


def score_no_answer(resp, valid: Set[str]) -> float:
    if not resp.refs:
        return 1.0
    for r in resp.refs:
        if get_file(r) in valid:
            return 0.2
    return 0.0


def score_offtopic(resp, valid: Set[str]) -> float:
    if resp.refs:
        for r in resp.refs:
            if get_file(r) in valid:
                return 0.0
        return 0.0
    return 1.0 if is_uncertain(resp.text) else 0.2


def evaluate(system, docs: List[str], show: bool = True) -> float:
    valid = get_all_files(docs)
    total = 0.0
    
    if show:
        print("Questions with answers:")
    for q in QUESTIONS:
        r = system.ask(q.text)
        s = score_answered(r, q, valid)
        ref_info = ""
        if r.refs:
            ref_info = f" -> {get_file(r.refs[0])}, p.{get_page(r.refs[0])}"
        if show:
            print(f"  [{s:.1f}] {q.text[:45]}...{ref_info}")
        total += s
    
    if show:
        print("Unanswerable:")
    for qt in UNANSWERABLE:
        r = system.ask(qt)
        s = score_no_answer(r, valid)
        if show:
            print(f"  [{s:.1f}] {qt[:50]}...")
        total += s
    
    if show:
        print("Off-topic:")
    for qt in OFFTOPIC:
        r = system.ask(qt)
        s = score_offtopic(r, valid)
        if show:
            print(f"  [{s:.1f}] {qt[:50]}...")
        total += s
    
    return total


def show_results(rag: float, tfidf: float, baseline: float):
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"  RAG:      {rag:.2f} / 10")
    print(f"  TF-IDF:   {tfidf:.2f} / 10")
    print(f"  LLM Only: {baseline:.2f} / 10")
    print("=" * 50)
    best = max([("RAG", rag), ("TF-IDF", tfidf), ("LLM", baseline)], key=lambda x: x[1])
    print(f"  Best: {best[0]}")
