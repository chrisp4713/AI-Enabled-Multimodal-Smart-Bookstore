
import json
from pathlib import Path
from datetime import datetime, timezone

# Locate database files
DATA = Path(__file__).resolve().parent.parent / "data"
BOOK_FILE = DATA / "book.json"
HISTORY_FILE = DATA / "view_history.json"


# Load all books
def load_books():
    with open(BOOK_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


# Search for a book by title
def get_book(title):
    books = load_books()

    for book in books:
        if book["Title"].lower() == title.lower():
            return book

    return None


# Return all books sorted by ID
def get_list():
    books = load_books()

    return sorted(books, key=lambda book: book["id"])


# Browse books by category, author, or publication date
def browse_book(category=None, author=None, date=None):
    books = load_books()
    results = []

    for book in books:
        match = True

        if category and category.lower() not in book["Category"].lower():
            match = False

        if author and author.lower() not in book["Author"].lower():
            match = False

        if date and not book["Pub_year"].startswith(str(date)):
            match = False

        if match:
            results.append(book)

    return sorted(results, key=lambda book: book["id"])


# Search keywords in book descriptions
def keyword_search(keyword):
    books = load_books()
    results = []

    for book in books:
        if keyword.lower() in book["Description"].lower():
            results.append(book)

    return sorted(results, key=lambda book: book["id"])


# Record browsing history
def log_view(book, user_id):
    if book is None or user_id is None:
        return

    # Load existing history
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            history = json.load(file)
    else:
        history = []

    # Create history record
    record = {
        "user_id": user_id,
        "book_id": book["id"],
        "Title": book["Title"],
        "Viewed_at": datetime.now(timezone.utc).isoformat()
    }

    history.append(record)

    # Save updated history
    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4, ensure_ascii=False)
