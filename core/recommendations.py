"""
Home page recommendations.                                          UC-03

    recommend(session, count=3) -> list of book dicts

    RC / VC  personalized: books in the categories and by the authors the
             user has browsed or bought, that they have not bought yet,
             most popular first. Topped up with the store-wide top books.
    VS / SU  the store-wide top books (most purchased, then most browsed).
    New users with no history also get the store-wide top books.

Reads browsing.csv (written by Jin's log_view) and purchases.csv.
"""
from __future__ import annotations

from collections import Counter

from core import store
from core.accounts import RC, VC, Session
from core.purchases import PURCHASES_FILE, list_books

BROWSING_FILE = "browsing.csv"
PURCHASE_WEIGHT = 3      # a purchase says more about taste than a view


def _popularity() -> Counter:
    """Store-wide score per book id: purchases count 3, views count 1."""
    score: Counter = Counter()
    for r in store.read_csv(PURCHASES_FILE):
        score[r["book_id"]] += PURCHASE_WEIGHT
    for r in store.read_csv(BROWSING_FILE):
        score[r["book_id"]] += 1
    return score


def top_books(count: int = 3, exclude: set[str] | None = None) -> list[dict]:
    """Most purchased and browsed books in the whole store."""
    exclude = exclude or set()
    books = [b for b in list_books() if str(b.get("id")) not in exclude]
    pop = _popularity()
    books.sort(key=lambda b: (-pop[str(b.get("id"))], str(b.get("title", ""))))
    return books[:count]


def recommend(session: Session, count: int = 3) -> list[dict]:
    if session.role not in (RC, VC) or not session.username:
        return top_books(count)

    user = session.username
    bought = {r["book_id"] for r in store.read_csv(PURCHASES_FILE) if r["user"] == user}
    viewed = [r["book_id"] for r in store.read_csv(BROWSING_FILE) if r["user"] == user]
    if not bought and not viewed:
        return top_books(count)

    books = {str(b.get("id")): b for b in list_books()}
    taste: Counter = Counter()
    for book_id, weight in [(b, PURCHASE_WEIGHT) for b in bought] + [(v, 1) for v in viewed]:
        book = books.get(book_id)
        if book:
            taste[("category", book.get("category"))] += weight
            taste[("author", book.get("author"))] += weight

    pop = _popularity()

    def match(book: dict) -> int:
        return (taste[("category", book.get("category"))]
                + taste[("author", book.get("author"))])

    candidates = [b for bid, b in books.items() if bid not in bought and match(b) > 0]
    candidates.sort(key=lambda b: (-match(b), -pop[str(b.get("id"))], str(b.get("title", ""))))
    picks = candidates[:count]

    if len(picks) < count:                      # top up with store favourites
        taken = bought | {str(b.get("id")) for b in picks}
        picks += top_books(count - len(picks), exclude=taken)
    return picks
