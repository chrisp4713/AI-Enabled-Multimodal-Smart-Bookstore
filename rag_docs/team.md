# Team RAG Documentation: "PBA Bookstore"

## Team Members & Titles

### 1. Alyssa Johnson — Frontend & Integration Lead
- **Module:** `app/` (`main_window.py`, `home_view.py`, `browse_view.py`, `book_view.py`, `account_view.py`, `login_view.py`, `chat_view.py`, `admin_views.py`)
- **Email:** ALYSSA.JOHNSON43@login.cuny.edu
- **Responsibilities:** Builds role-based navigation (VS, RC, VC, SU), all UI screens (home, browse/search, book detail, account, login/registration, chat, and SU admin views), handles Ollama/camera fallback alerts, and leads the midpoint demo and integration check.

### 2. Christopher Persaud — AI Biometrics & Computer Vision Engineer
- **Module:** `ai/` (`biometrics.py`, `vision.py`)
- **Email:** CHRISTOPHER.PERSAUD31@stu-mail.ccny.cuny.edu
- **Responsibilities:** Implements local face and voice enrollment and authentication, camera/mic capture, and JPG/PNG cover image indexing and visual search (`index_images`, `search_by_image` within 5s).

### 3. Sahadat Amzad — Backend Engineer
- **Module:** `core/` (`store.py`, `accounts.py`, `purchases.py`, `recommendations.py`)
- **Email:** Sahadat.Amzad@cuny.edu
- **Responsibilities:** Manages atomic JSON file writes, seed users, user account lifecycle, purchases, warning tracking, SU reactivation, top-3 book recommendations, and SU finance report data from `purchases.csv`.

### 4. Jin Chao — Backend & Catalog Data Engineer
- **Module:** `catalog/` (`catalog.py`, `community.py`) and `data/` (`books.json`)
- **Email:** JINCHAO.CHEN30@stu-mail.ccny.cuny.edu
- **Responsibilities:** Curates 10–20 public-domain books across 3+ categories, builds catalog browsing and keyword search with view logging, manages 1–5 star weighted ratings, enforces taboo-word masking and comment blocking/warnings, updates to-order/no-order lists, and handles book addition and SU taboo-word management.

### 5. Jessica Barrera Saguay — LLM & RAG Systems Engineer / Documentation Lead
- **Module:** `llm/` (`chat.py`, `rag.py`), `rag_docs/`, and `docs/`
- **Email:** JESSICA.BARRERASAGUAY81@stu-mail.ccny.cuny.edu
- **Responsibilities:** Configures local Ollama model, authors `rag_docs/` store and team documentation, implements role-based Store Chat and chapter-grounded Ask the Book (`index_book`, `ask_book` with ownership checks, spoiler prompts, and longer VC responses), and leads the SRS final pass, README, and 12 test scenarios.

## Communication & Repository
- **Communication Channel:** Discord (used for team coordination and progress updates)
- **Code Repository:** GitHub (`https://github.com/chrisp4713/AI-Enabled-Multimodal-Smart-Bookstore`)

## Workflow & Code Standards
- **Module Ownership:** Each team member edits their assigned directory to prevent merge conflicts.
- **Traceability:** Every file and function includes Use Case comments (`UC-01` through `UC-14`) at the top for easier access.
