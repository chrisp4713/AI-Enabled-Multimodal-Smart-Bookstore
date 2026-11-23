"""
Accounts: roles, sessions, deposits, warnings and reactivation.
                                            UC-06, UC-12; BR-02, BR-03, BR-09

Roles
-----
    VS  visitor (nobody logged in; never stored in users.json)
    RC  registered customer
    VC  VIP customer (10% discount, see purchases.py)
    SU  super user (store manager)

users.json format (data contract)
---------------------------------
    { "<username>": { "username", "display_name", "role", "balance",
                      "warnings", "purchase_count", "total_spent", "status" } }
    status is "active" or "suspended".

Public functions
----------------
    Session, VISITOR                    who is logged in (passed to screens)
    create_account(username, display_name)            -> User   (UC-01)
    get_user(username)                                  -> User | None
    is_active(username)                                 -> bool   (used by login)
    has_purchased(username, book_id)                    -> bool   (Ask the Book, ratings)
    deposit(username, amount)                           -> User   (UC-06)
    purchase_history(username)                          -> list[dict]
    apply_warning(username)                             -> WarningResult (BR-09)
    suspend(username, reason)                           -> User   (BR-02)
    list_suspended()                                    -> list[User] (UC-12)
    reactivate(actor, username, deposit_amount)         -> User   (UC-12, BR-03)
    seed_demo_users()                                   one user per role
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from core import store

USERS_FILE = "users.json"
PURCHASES_FILE = "purchases.csv"

VS, RC, VC, SU = "VS", "RC", "VC", "SU"
ACTIVE, SUSPENDED = "active", "suspended"
MAX_WARNINGS = 2        # more than this triggers BR-09


class AccountError(Exception):
    """A rule was broken; the message is safe to show to the user."""


# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------
@dataclass
class User:
    username: str
    display_name: str
    role: str = RC
    balance: float = 0.0
    warnings: int = 0
    purchase_count: int = 0
    total_spent: float = 0.0
    status: str = ACTIVE
    suspension_reason: str = ""

    @property
    def is_active(self) -> bool:
        return self.status == ACTIVE


@dataclass(frozen=True)
class Session:
    """The logged-in user, passed around by the GUI. VISITOR when nobody is."""
    username: str | None = None
    role: str = VS

    @property
    def is_visitor(self) -> bool:
        return self.role == VS

    @classmethod
    def for_user(cls, user: "User") -> "Session":
        return cls(user.username, user.role)


VISITOR = Session()


@dataclass
class WarningResult:
    user: User
    action: str          # "warned", "suspended" or "demoted"
    message: str


# --------------------------------------------------------------------------
# Storage helpers
# --------------------------------------------------------------------------
def _load_all() -> dict[str, dict]:
    return store.load_json(USERS_FILE, default={})


def _save_user(user: User) -> None:
    users = _load_all()
    users[user.username] = asdict(user)
    store.save_json(USERS_FILE, users)


def _require(username: str) -> User:
    user = get_user(username)
    if user is None:
        raise AccountError(f"No account named '{username}'.")
    return user


def _money(x: float) -> float:
    return round(float(x), 2)


# --------------------------------------------------------------------------
# Accounts (UC-01 creation, UC-06 deposits)
# --------------------------------------------------------------------------
def create_account(username: str, display_name: str = "") -> User:
    """New customer: RC, $0.00, zero warnings, zero purchases (UC-01)."""
    username = (username or "").strip()
    if not username:
        raise AccountError("Username cannot be empty.")
    if get_user(username) is not None:
        raise AccountError(f"The username '{username}' is already taken.")
    user = User(username=username, display_name=(display_name or username).strip())
    _save_user(user)
    return user


def get_user(username: str) -> User | None:
    record = _load_all().get(username)
    if record is None:
        return None
    known = User.__dataclass_fields__.keys()
    return User(**{k: v for k, v in record.items() if k in known})


def all_users() -> list[User]:
    return [u for u in (get_user(name) for name in _load_all()) if u is not None]


def is_active(username: str) -> bool:
    """False for unknown or suspended accounts (login is refused)."""
    user = get_user(username)
    return user is not None and user.is_active


def deposit(username: str, amount: float) -> User:
    """Add money to an account. Only positive amounts are accepted (UC-06)."""
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        raise AccountError("Deposit amount must be a number.") from None
    if amount <= 0:
        raise AccountError("Deposit amount must be greater than $0.00.")
    user = _require(username)
    user.balance = _money(user.balance + amount)
    _save_user(user)
    return user


def has_purchased(username: str, book_id) -> bool:
    book_id = str(book_id)
    return any(r["user"] == username and r["book_id"] == book_id
               for r in store.read_csv(PURCHASES_FILE))


def purchase_history(username: str) -> list[dict]:
    """The user's purchases, newest first (shown on My Account, UC-06)."""
    rows = [r for r in store.read_csv(PURCHASES_FILE) if r["user"] == username]
    return sorted(rows, key=lambda r: r["date"], reverse=True)


# --------------------------------------------------------------------------
# Warnings and suspension (BR-02, BR-09)
# --------------------------------------------------------------------------
def suspend(username: str, reason: str = "") -> User:
    user = _require(username)
    user.status = SUSPENDED
    user.suspension_reason = reason
    _save_user(user)
    return user


def apply_warning(username: str) -> WarningResult:
    """Give a warning (e.g. a blocked comment, BR-08) and apply BR-09:
    an RC with more than 2 warnings is suspended; a VC with more than 2
    warnings is demoted to RC with warnings reset to 0."""
    user = _require(username)
    user.warnings += 1

    if user.warnings > MAX_WARNINGS and user.role == RC:
        user.status = SUSPENDED
        user.suspension_reason = "Too many warnings."
        _save_user(user)
        return WarningResult(user, "suspended",
                             "Your account has been suspended after more than 2 warnings.")

    if user.warnings > MAX_WARNINGS and user.role == VC:
        user.role = RC
        user.warnings = 0
        _save_user(user)
        return WarningResult(user, "demoted",
                             "You received more than 2 warnings and lost VIP status.")

    _save_user(user)
    return WarningResult(user, "warned",
                         f"You received a warning ({user.warnings} so far).")


# --------------------------------------------------------------------------
# Super user: reopen suspended accounts (UC-12, BR-03)
# --------------------------------------------------------------------------
def list_suspended() -> list[User]:
    return [u for u in all_users() if u.status == SUSPENDED]


def reactivate(actor: Session, username: str, deposit_amount: float = 0.0) -> User:
    """SU records a deposit for a suspended user. The account is reopened
    only if the balance is then above $0 (BR-03)."""
    if actor.role != SU:
        raise AccountError("Only the store manager can reactivate accounts.")
    user = _require(username)
    if deposit_amount:
        user = deposit(username, deposit_amount)
    if user.balance <= 0:
        raise AccountError(
            f"Balance is ${user.balance:.2f}. It must be above $0.00 to reactivate.")
    user.status = ACTIVE
    user.suspension_reason = ""
    _save_user(user)
    return user


# --------------------------------------------------------------------------
# Demo data
# --------------------------------------------------------------------------
def seed_demo_users(overwrite: bool = False) -> None:
    """Create one user per stored role so every screen can be tried."""
    if _load_all() and not overwrite:
        return
    demo = [
        User("manager", "Store Manager", role=SU),
        User("vip_vera", "Vera (VIP)", role=VC, balance=150.0),
        User("reader_ray", "Ray (Customer)", role=RC, balance=50.0),
    ]
    store.save_json(USERS_FILE, {u.username: asdict(u) for u in demo})


if __name__ == "__main__":
    seed_demo_users(overwrite=True)
    for u in all_users():
        print(u)
