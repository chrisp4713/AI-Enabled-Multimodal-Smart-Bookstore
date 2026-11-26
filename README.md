# AI-Enabled Multimodal Smart Bookstore

An AI-enabled, multimodal bookstore application built for our Software Engineering course at The City College of New York (CCNY). The platform combines role-based book purchasing, catalog management, biometric authentication, image-based book search, and a local LLM/RAG assistant for store inquiries and chapter-grounded book Q&A.

---

## Team & Responsibilities

| Team Member | Module |
|---|---|
| **Alyssa Johnson** | Frontend UI & System Integration (`app/`) |
| **Christopher Persaud** | Face/voice login & Picture Search (`ai/`) |
| **Sahadat Amzad** | Backend Accounts, Purchases & Finance Reporting (`core/`) |
| **Jin Chao** | Backend Catalog, Ratings & Admin Lists (`catalog/`) |
| **Jessica Barrera Saguay** | LLM/RAG Chat Assistant, Book Q&A & Documentation (`llm/`, `rag_docs/`) |

---

## Key Features

* **Role-Based Access Control:** Supports Visitor (`VS`), Regular Customer (`RC`), VIP Customer (`VC`), and Super User (`SU`) roles with tailored permissions and chat styles.
* **Store Chat Assistant (UC-09):** Uses `all-MiniLM-L6-v2` embeddings and cosine similarity (threshold `0.25`) to route store/team questions to `[RAG]` (`rag_docs/`) and general questions to `[LLM]` (local Ollama `gemma2:2b`).
* **SU-Only Finance Reporting (UC-09):** Blocks financial queries for `VS`, `RC`, and `VC` users and serves live purchase/revenue summaries exclusively to `SU` users.
* **Chapter-Grounded Book Q&A (UC-14):** Splits owned books by chapter, caches embeddings in `book_index/`, cites chapters in answers, enforces spoiler protection, and provides deeper responses for `VC` members.
* **Multimodal & Core Storefront:** Includes biometric login, picture-based book search, catalog browsing, ratings, purchasing with VIP discounts, and admin management.

---

## Prerequisites

* **Python 3.10+**
* **Ollama** installed locally ([ollama.com](https://ollama.com))
* The `gemma2:2b` model pulled locally:
  ```bash
  ollama pull gemma2:2b
