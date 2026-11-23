"""Tests for core/ (accounts, purchases, recommendations). Run: pytest tests/

Covers test scenarios 3 (deposit and buy), 4 (overdraft), 5 (reopen),
6 (VIP), the BR-09 part of 7, and the no-order block from 8.
"""
import pytest

from core import store
from core import accounts as acc
from core import purchases as pur
from core.recommendations import recommend, top_books

BOOKS = [
    {"id": "b1", "title": "Moby Dick", "author": "Melville", "category": "literature", "price": 20.0},
    {"id": "b2", "title": "Origin of Species", "author": "Darwin", "category": "science", "price": 30.0},
    {"id": "b3", "title": "Billy Budd", "author": "Melville", "category": "literature", "price": 10.0},
    {"id": "b4", "title": "Relativity", "author": "Einstein", "category": "science", "price": 60.0},
    {"id": "b5", "title": "Emma", "author": "Austen", "category": "literature", "price": 15.0},
]


@pytest.fixture(autouse=True)
def tmp_store(tmp_path):
    store.set_data_dir(tmp_path)
    store.save_json("books.json", BOOKS)
    store.save_json("order_lists.json", {"to_order": [], "no_order": []})
    acc.seed_demo_users(overwrite=True)
    yield


# --- store ---------------------------------------------------------------
def test_atomic_json_roundtrip_leaves_no_temp_files(tmp_path):
    store.save_json("x.json", {"a": 1})
    assert store.load_json("x.json") == {"a": 1}
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


# --- accounts ------------------------------------------------------------
def test_new_account_starts_empty():
    u = acc.create_account("newbie", "New Bie")
    assert (u.role, u.balance, u.warnings, u.purchase_count) == ("RC", 0.0, 0, 0)
    with pytest.raises(acc.AccountError):
        acc.create_account("newbie")


@pytest.mark.parametrize("amount", [0, -5, "abc"])
def test_deposit_rejects_non_positive(amount):
    with pytest.raises(acc.AccountError):
        acc.deposit("reader_ray", amount)


# --- scenario 3: deposit and buy -------------------------------------------
def test_deposit_and_buy():
    acc.create_account("sam")
    acc.deposit("sam", 50)
    r = pur.buy_book("sam", "b1")
    assert r.ok and r.new_balance == 30.0
    assert acc.has_purchased("sam", "b1")
    assert acc.purchase_history("sam")[0]["book_id"] == "b1"


# --- scenario 4 + 5: overdraft, suspension, reopen -------------------------
def test_overdraft_needs_confirmation_then_fines_and_suspends():
    acc.create_account("sam")
    acc.deposit("sam", 10)
    first = pur.buy_book("sam", "b2")
    assert not first.ok and first.needs_confirmation
    r = pur.buy_book("sam", "b2", confirm_overdraft=True)
    assert r.ok and r.fine == 1.0 and r.new_balance == -21.0
    assert r.suspended and r.logged_out
    assert not acc.is_active("sam")


def test_reactivate_only_su_and_only_above_zero():
    acc.create_account("sam")
    pur.buy_book("sam", "b1", confirm_overdraft=True)          # balance -21
    assert [u.username for u in acc.list_suspended()] == ["sam"]
    su = acc.Session("manager", acc.SU)
    with pytest.raises(acc.AccountError):
        acc.reactivate(acc.Session("reader_ray", acc.RC), "sam", 100)
    with pytest.raises(acc.AccountError):
        acc.reactivate(su, "sam", 21)                            # lands on $0.00
    assert not acc.is_active("sam")
    acc.reactivate(su, "sam", 5)                                # now $5.00
    assert acc.is_active("sam")


# --- scenario 6: VIP --------------------------------------------------------
def test_fourth_purchase_promotes_and_next_gets_discount():
    acc.create_account("sam")
    acc.deposit("sam", 500)
    for book in ["b1", "b3", "b5"]:
        assert not pur.buy_book("sam", book).promoted_to_vip
    assert pur.buy_book("sam", "b2").promoted_to_vip           # 4th purchase
    q = pur.quote("sam", "b4")
    assert q.discount == 6.0 and q.final_price == 54.0


def test_spending_over_200_promotes():
    store.save_json("books.json", BOOKS + [{"id": "big", "title": "Atlas", "author": "X",
                                            "category": "science", "price": 205.0}])
    acc.create_account("sam")
    acc.deposit("sam", 300)
    assert pur.buy_book("sam", "big").promoted_to_vip


# --- scenario 7: warnings (BR-09) ------------------------------------------
def test_third_warning_suspends_rc():
    acc.create_account("sam")
    assert acc.apply_warning("sam").action == "warned"
    assert acc.apply_warning("sam").action == "warned"
    assert acc.apply_warning("sam").action == "suspended"
    assert not acc.is_active("sam")


def test_third_warning_demotes_vc_and_resets():
    for _ in range(2):
        acc.apply_warning("vip_vera")
    res = acc.apply_warning("vip_vera")
    assert res.action == "demoted"
    u = acc.get_user("vip_vera")
    assert (u.role, u.warnings, u.status) == ("RC", 0, "active")


# --- scenario 8: no-order list blocks buying --------------------------------
def test_no_order_book_cannot_be_bought():
    store.save_json("order_lists.json", {"to_order": [], "no_order": ["b1"]})
    q = pur.quote("reader_ray", "b1")
    assert not q.can_buy
    assert not pur.buy_book("reader_ray", "b1").ok


def test_visitor_and_su_cannot_buy():
    assert not pur.quote(None, "b1").can_buy
    assert not pur.quote("manager", "b1").can_buy


# --- finance report (UC-09) -------------------------------------------------
def test_finance_report_su_only():
    pur.buy_book("reader_ray", "b1")
    with pytest.raises(acc.AccountError):
        pur.finance_report(acc.RC)
    r = pur.finance_report(acc.SU)
    assert r["purchases"] == 1 and r["revenue"] == 20.0
    assert "Moby Dick" in pur.finance_report_text(acc.SU)


# --- recommendations (UC-03) ------------------------------------------------
def test_visitor_gets_store_top_three():
    pur.buy_book("reader_ray", "b4", confirm_overdraft=True)
    store.append_csv("browsing.csv", {"user": "x", "book_id": "b2", "timestamp": "t"},
                     ["user", "book_id", "timestamp"])
    top = [b["id"] for b in recommend(acc.VISITOR)]
    assert top[:2] == ["b4", "b2"] and len(top) == 3


def test_customer_gets_personalized_books():
    acc.deposit("reader_ray", 100)
    pur.buy_book("reader_ray", "b1")                            # Melville, literature
    picks = [b["id"] for b in recommend(acc.Session("reader_ray", acc.RC))]
    assert "b1" not in picks                                    # already owned
    assert picks[0] == "b3"                                     # same author + category
    assert len(picks) == 3


def test_new_customer_falls_back_to_top():
    acc.create_account("fresh")
    assert recommend(acc.Session("fresh", acc.RC)) == top_books(3)
