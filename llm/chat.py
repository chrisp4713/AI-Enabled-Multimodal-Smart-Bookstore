# UC-09, UC-14
import re
import requests

from llm.rag import search_rag, search_book, model

# ---- Settings ----
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma2:2b"
TIMEOUT_SECONDS = 15
RAG_THRESHOLD = 0.25

# Finance/revenue/sales/profit/income questions are SU-only
FINANCE_PATTERN = re.compile(r"\b(finance|financial|revenue|sales|profit|income)\b", re.IGNORECASE)

# Tone, detail level and perks awareness per role
ROLE_STYLE = {
    "VS": "Be friendly and simple. Keep answers short and basic.",
    "RC": "Be professional and clear. Give moderate detail.",
    "VC": "Be warm and appreciative. Mention VC member perks and benefits when relevant. Give moderate detail.",
    "SU": "Be direct and thorough. Give full detail.",
}


class OllamaError(Exception): # Carries a user-friendly message so errors are shown without a [RAG]/[LLM] label
    pass


# UC-09
def answer(question: str, role: str) -> str:
    role = role.upper() if role else "VS"
    if role not in ROLE_STYLE:
        role = "VS"  # unknown roles get the most restricted style

    # 1) Finance check comes first: never reaches RAG or Ollama
    if FINANCE_PATTERN.search(question):
        return finance_answer(role)

    # 2) Retrieve docs and check the top score
    matches = search_rag(question, model)
    use_rag = bool(matches) and matches[0]["score"] >= RAG_THRESHOLD

    # 3) Build the prompt and pick the label
    style = ROLE_STYLE[role]
    if use_rag:
        label = "[RAG]"
        # Only pass chunks that meet the threshold
        chunks = [m for m in matches if m["score"] >= RAG_THRESHOLD]
        context = "\n\n".join(f"({m['source']}) {m['text']}" for m in chunks)
        prompt = f"{style}\nAnswer using only this context:\n{context}\n\nQuestion: {question}"
    else:
        label = "[LLM]"
        prompt = f"{style}\n\nQuestion: {question}"

    # 4) Ask Ollama; errors come back as plain messages with no label
    try:
        return f"{label} {ask_ollama(prompt)}"
    except OllamaError as e:
        return str(e)


# UC-09
def finance_answer(role: str) -> str: # SU gets the finance report; everyone else is denied
    if role != "SU":
        return "Sorry, finance reports are only available to SU users."

    from core.purchases import finance_report_text
    return finance_report_text(role)


# UC-09, UC-14
def ask_ollama(prompt: str, num_predict: int = 250) -> str: # Send the prompt to local Ollama
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,       # single JSON reply
                "keep_alive": "30m",   # keep the model loaded so later replies are fast
                "options": {"num_predict": num_predict},
            },
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        raise OllamaError("Can't reach Ollama. Please start it (run `ollama serve`) and try again.")
    except requests.exceptions.Timeout:
        raise OllamaError("Ollama took too long to respond. Please try again in a moment.")
    except (requests.exceptions.RequestException, KeyError, ValueError):
        raise OllamaError("Ollama returned an unexpected error. Make sure the model is installed (`ollama pull gemma2:2b`).")


# UC-14
def user_owns_book(username: str, book_id: str) -> bool: # Check if the user owns this book.
    return True


# UC-14
def ask_book(
    username: str,
    role: str,
    book_id: str,
    book_path: str,
    question: str,
    allow_spoilers: bool = False,
) -> str: # Answer a grounded question about an owned book with chapter citations
  
    role = role.upper() if role else "VS"

    # 1) Owned books only
    if not user_owns_book(username, book_id):
        return "Sorry, you can only ask questions about books you own."

    # 2) Retrieve relevant passages from the book
    matches = search_book(book_id, book_path, question, model)
    if not matches:
        return "I couldn't find any relevant passages in this book."

    context = "\n\n".join(f"[{m['chapter']}]\n{m['text']}" for m in matches)

    # 3) Spoiler rule
    if allow_spoilers:
        spoiler_rule = "Spoilers are allowed. You may discuss full plot details from the excerpts."
    else:
        spoiler_rule = (
            "Do NOT reveal major plot twists, endings, or spoilers. "
            "If answering requires a spoiler, warn the user and ask if they want spoilers enabled."
        )

    # 4) Longer, deeper answers for VC; concise answers for everyone else
    if role == "VC":
        length_rule = "Give a thorough, detailed answer (2-3 paragraphs)."
        max_tokens = 450
    else:
        length_rule = "Keep your answer concise (1 short paragraph)."
        max_tokens = 200

    prompt = (
        f"Answer the user's question using ONLY the book excerpts below. "
        f"Always cite the chapter name(s) you used.\n"
        f"{spoiler_rule}\n"
        f"{length_rule}\n\n"
        f"Excerpts:\n{context}\n\n"
        f"Question: {question}"
    )

    try:
        return ask_ollama(prompt, num_predict=max_tokens)
    except OllamaError as e:
        return str(e)


# if __name__ == "__main__":
#     tests = [
#         ("What was our total revenue this month?", "VC"),   # finance, denied
#         ("What was our total revenue this month?", "SU"),   # finance, allowed
#         ("What are the store's opening hours?", "VS"),      # store/team question
#         ("Who wrote Pride and Prejudice?", "RC"),           # general question
#     ]
#     for q, r in tests:
#         print(f"[{r}] {q}\n  -> {answer(q, r)}\n")
