"""
Purchases: price quote, buying, overdraft, VIP promotion, finance report.
                                    UC-07, UC-09 (finance); BR-01, 02, 04, 05

Buy flow for the book screen (Alyssa)
-------------------------------------
    q = quote(session.username, book_id)        show q.final_price, q.discount
    if not q.can_buy: show q.message, disable Buy (no-order list, visitor...)
    if q.exceeds_balance: show q.message and ask "Continue anyway?"
    result = buy_book(session.username, book_id, confirm_overdraft=<answer>)
    if result.logged_out: return to the visitor screen (BR-02)
    show result.message

Rules
-----
    BR-05  VCs pay 10% less.
    BR-01  A purchase above the balance still goes through after a warning,
           and a $1.00 fine is added.
    BR-02  The account is then suspended and the user is logged out.
    BR-04  After a purchase, an RC with more than 3 purchases or more than
           $200.00 spent in total becomes a VC.
    No-order list: books on it cannot be bought (Jin's community.py keeps it).

purchases.csv columns: user, book_id, price_paid, discount, fine, date
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from core import store
from core.accounts import (RC, SU, VC, AccountError, User, _money, _save_user,
                           get_user)

PURCHASES_FILE = "purchases.csv"
PURCHASE_FIELDS = ["user", "book_id", "price_paid", "discount", "fine", "date"]
BOOKS_FILE = "books.json"
ORDER_LISTS_FILE = "order_lists.json"

VC_DISCOUNT = 0.10          # BR-05
OVERDRAFT_FINE = 1.00       # BR-01
VIP_MIN_PURCHASES = 3       # BR-04: more than 3 purchases ...
VIP_MIN_SPENT = 200.00      # BR-04: ... or more than $200 in total


# --------------------------------------------------------------------------
# Catalog access. Uses Jin's catalog.get_book / list_books once they exist,
# and reads data/books.json directly until then.
# --------------------------------------------------------------------------
def get_book(book_id) -> dict | None:
    try:
        from catalog.catalog import get_book as catalog_get_book
        return catalog_get_book(book_id)
    except ImportError:
        return next((b for b in list_books() if str(b.get("id")) == str(book_id)), None)


def list_books() -> list[dict]:
    try:
        from catalog.catalog import list_books as catalog_list_books
        return list(catalog_list_books())
    except ImportError:
        books = store.load_json(BOOKS_FILE, default=[])
        return list(books.values()) if isinstance(books, dict) else list(books)


def is_on_no_order_list(book_id) -> bool:
    lists = store.load_json(ORDER_LISTS_FILE, default={}) or {}
    return str(book_id) in {str(b) for b in lists.get("no_order", [])}


# --------------------------------------------------------------------------
# Quote and buy
# --------------------------------------------------------------------------
@dataclass
class Quote:
    book_id: str
    list_price: float
    discount: float
    final_price: float
    can_buy: bool
    exceeds_balance: bool
    message: str


@dataclass
class PurchaseResult:
    ok: bool
    message: str                     # safe to show to the user
    price_paid: float = 0.0
    fine: float = 0.0
    new_balance: float = 0.0
    promoted_to_vip: bool = False
    suspended: bool = False
    logged_out: bool = False
    needs_confirmation: bool = False


def quote(username: str | None, book_id) -> Quote:
    """What the user would pay, and whether they may buy at all."""
    book = get_book(book_id)
    if book is None:
        return Quote(str(book_id), 0.0, 0.0, 0.0, False, False, "Book not found.")
    price = _money(book.get("price", 0))
    user = get_user(username) if username else None

    def blocked(msg):
        return Quote(str(book_id), price, 0.0, price, False, False, msg)

    if user is None:
        return blocked("Please log in to buy books.")
    if not user.is_active:
        return blocked("Your account is suspended.")
    if user.role == SU:
        return blocked("The store manager account cannot buy books.")
    if is_on_no_order_list(book_id):
        return blocked("This book is no longer available for purchase.")

    discount = _money(price * VC_DISCOUNT) if user.role == VC else 0.0
    final = _money(price - discount)
    over = final > user.balance
    if over:
        msg = (f"This costs ${final:.2f} but your balance is ${user.balance:.2f}. "
               f"If you continue, a ${OVERDRAFT_FINE:.2f} fine is added, your "
               f"account will be suspended and you will be logged out.")
    else:
        msg = f"Buy for ${final:.2f}?"
    return Quote(str(book_id), price, discount, final, True, over, msg)


def buy_book(username: str, book_id, confirm_overdraft: bool = False) -> PurchaseResult:
    """Buy one book (UC-07). Call quote() first to show the price."""
    q = quote(username, book_id)
    if not q.can_buy:
        return PurchaseResult(False, q.message)
    if q.exceeds_balance and not confirm_overdraft:
        return PurchaseResult(False, q.message, needs_confirmation=True)

    user = get_user(username)
    fine = OVERDRAFT_FINE if q.exceeds_balance else 0.0           # BR-01
    user.balance = _money(user.balance - q.final_price - fine)
    user.purchase_count += 1
    user.total_spent = _money(user.total_spent + q.final_price)

    promoted = _check_vip(user)                                  # BR-04
    suspended = q.exceeds_balance                                # BR-02
    if suspended:
        user.status = "suspended"
        user.suspension_reason = "Negative balance after a purchase."
    _save_user(user)

    store.append_csv(PURCHASES_FILE, {
        "user": username,
        "book_id": q.book_id,
        "price_paid": f"{q.final_price:.2f}",
        "discount": f"{q.discount:.2f}",
        "fine": f"{fine:.2f}",
        "date": datetime.now().isoformat(timespec="seconds"),
    }, PURCHASE_FIELDS)

    parts = [f"Purchased for ${q.final_price:.2f}."]
    if fine:
        parts.append(f"A ${fine:.2f} fine was added. Your balance is "
                     f"${user.balance:.2f} and your account is suspended.")
    if promoted:
        parts.append("Congratulations, you are now a VIP customer (10% off future books)!")
    return PurchaseResult(True, " ".join(parts), q.final_price, fine, user.balance,
                          promoted, suspended, logged_out=suspended)


def _check_vip(user: User) -> bool:
    """BR-04: promote an RC to VC after the purchase that crosses a limit."""
    if user.role != RC:
        return False
    if user.purchase_count > VIP_MIN_PURCHASES or user.total_spent > VIP_MIN_SPENT:
        user.role = VC
        return True
    return False


# --------------------------------------------------------------------------
# Finance report for the SU chat (UC-09, used by Jessica's chat.py)
# --------------------------------------------------------------------------
def finance_report(actor_role: str) -> dict:
    """Store totals from purchases.csv. Only the SU may see this."""
    if actor_role != SU:
        raise AccountError("Only the store manager can see the finance report.")
    rows = store.read_csv(PURCHASES_FILE)
    revenue = sum(float(r["price_paid"]) for r in rows)
    fines = sum(float(r["fine"]) for r in rows)
    discounts = sum(float(r["discount"]) for r in rows)
    sales: dict[str, int] = {}
    for r in rows:
        sales[r["book_id"]] = sales.get(r["book_id"], 0) + 1
    top = sorted(sales.items(), key=lambda kv: kv[1], reverse=True)[:5]
    titles = {str(b.get("id")): b.get("title", "") for b in list_books()}
    return {
        "purchases": len(rows),
        "revenue": _money(revenue),
        "fines_collected": _money(fines),
        "discounts_given": _money(discounts),
        "total_income": _money(revenue + fines),
        "customers": len({r["user"] for r in rows}),
        "top_books": [{"book_id": b, "title": titles.get(b, ""), "copies": n} for b, n in top],
    }


def finance_report_text(actor_role: str) -> str:
    """Plain-text version of finance_report() for the chat window."""
    r = finance_report(actor_role)
    lines = [
        f"Purchases: {r['purchases']} by {r['customers']} customers",
        f"Book revenue: ${r['revenue']:.2f}",
        f"Fines collected: ${r['fines_collected']:.2f}",
        f"VIP discounts given: ${r['discounts_given']:.2f}",
        f"Total income: ${r['total_income']:.2f}",
    ]
    if r["top_books"]:
        lines.append("Top sellers: " + ", ".join(
            f"{b['title'] or b['book_id']} ({b['copies']})" for b in r["top_books"]))
    return "\n".join(lines)
