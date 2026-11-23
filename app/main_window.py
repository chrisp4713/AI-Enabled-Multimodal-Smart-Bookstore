"""Main window with navigation that changes by role.  (UC-01 to UC-14 entry point)

Roles:
    VS = visitor (not logged in)
    RC = regular customer
    VC = VIP customer
    SU = superuser (store admin)

How it works
    - SCREENS lists every screen once: key, sidebar label, where the view class lives, UC number.
    - NAV_BY_ROLE says which screen keys each role sees in the sidebar.
    - Each view is built the first time it is opened. If its file or class does not exist yet,
      a placeholder is shown, so the window runs before every screen is written.

Convention for every view class in app/*.py:
    class SomeView(QWidget):
        def __init__(self, window): ...        # window is this MainWindow
        def on_show(self, user): ...           # optional, called every time the screen is opened

    Views change who is logged in with  window.set_user(user)  and
    open a book with  window.show_book(book_id).
"""
from dataclasses import dataclass
from importlib import import_module

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

# Set to False before the final demo to hide the role switcher.
DEV_MODE = True

VS, RC, VC, SU = "VS", "RC", "VC", "SU"
ROLE_NAMES = {
    VS: "Visitor",
    RC: "Regular customer",
    VC: "VIP customer",
    SU: "Superuser",
}


@dataclass
class CurrentUser:
    """What the GUI needs to know about whoever is logged in.

    Later, build this from Sahadat's get_user() result instead of by hand.
    """
    username: str = "visitor"
    role: str = VS


VISITOR = CurrentUser()


@dataclass(frozen=True)
class Screen:
    key: str
    label: str
    module: str      # file in app/
    class_name: str  # view class inside that file
    uc: str          # use case(s), shown on placeholders


SCREENS = {
    s.key: s
    for s in [
        Screen("home", "Home", "app.home_view", "HomeView", "UC-03"),
        Screen("browse", "Browse & Search", "app.browse_view", "BrowseView", "UC-04, UC-05"),
        Screen("book", "Book", "app.book_view", "BookView", "UC-07, UC-08"),  # opened from Home/Browse
        Screen("account", "My Account", "app.account_view", "AccountView", "UC-06"),
        Screen("login", "Log in", "app.login_view", "LoginView", "UC-02"),
        Screen("register", "Register", "app.login_view", "RegisterView", "UC-01"),
        Screen("store_chat", "Store Chat", "app.chat_view", "StoreChatView", "UC-09"),
        Screen("ask_book", "Ask the Book", "app.chat_view", "AskBookView", "UC-14"),
        Screen("suspended", "Suspended Accounts", "app.admin_views", "SuspendedView", "UC-12"),
        Screen("add_book", "Add Book", "app.admin_views", "AddBookView", "UC-10"),
        Screen("taboo", "Taboo Words", "app.admin_views", "TabooWordsView", "UC-11"),
        Screen("order_lists", "To-Order / No-Order", "app.admin_views", "OrderListsView", "UC-13"),
    ]
}

# What each role sees in the sidebar, in order. "logout" is an action, not a screen.
NAV_BY_ROLE = {
    VS: ["home", "browse", "login", "register", "store_chat"],
    RC: ["home", "browse", "account", "store_chat", "ask_book", "logout"],
    VC: ["home", "browse", "account", "store_chat", "ask_book", "logout"],
    SU: ["home", "browse", "suspended", "add_book", "taboo", "order_lists", "store_chat", "logout"],
}

LOGOUT_LABEL = "Log out"


class PlaceholderView(QWidget):
    """Shown when a screen's real view is not written yet (or fails to load)."""

    def __init__(self, window, screen: Screen, error: str = ""):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel(screen.label)
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        title.setAlignment(Qt.AlignCenter)

        detail = QLabel(f"{screen.uc}\n{screen.module}.{screen.class_name} not built yet")
        detail.setAlignment(Qt.AlignCenter)
        detail.setStyleSheet("color: gray;")

        layout.addWidget(title)
        layout.addWidget(detail)

        if error:  # a real view exists but crashed while loading
            err = QLabel(f"Could not load this screen:\n{error}")
            err.setAlignment(Qt.AlignCenter)
            err.setWordWrap(True)
            err.setStyleSheet("color: #b3261e;")
            layout.addWidget(err)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Bookstore")
        self.resize(1100, 720)

        self.user = VISITOR
        self._views = {}  # screen key -> built view, so each is created once

        # --- layout: sidebar on the left, stacked screens on the right ---
        root = QWidget()
        self.setCentralWidget(root)
        row = QHBoxLayout(root)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        side_layout = QVBoxLayout(sidebar)

        self.user_label = QLabel()
        self.user_label.setWordWrap(True)
        self.user_label.setStyleSheet("font-weight: 600; padding: 6px;")
        side_layout.addWidget(self.user_label)

        self.nav = QListWidget()
        self.nav.setFrameShape(QListWidget.NoFrame)
        self.nav.currentItemChanged.connect(self._on_nav_changed)
        side_layout.addWidget(self.nav, stretch=1)

        if DEV_MODE:
            side_layout.addWidget(QLabel("DEV: switch role"))
            self.dev_roles = QComboBox()
            for role in (VS, RC, VC, SU):
                self.dev_roles.addItem(f"{role} - {ROLE_NAMES[role]}", role)
            self.dev_roles.currentIndexChanged.connect(self._on_dev_role_changed)
            side_layout.addWidget(self.dev_roles)

        self.stack = QStackedWidget()

        row.addWidget(sidebar)
        row.addWidget(self.stack, stretch=1)

        self.set_user(VISITOR)

    # ---------- public API for the views ----------

    def set_user(self, user: CurrentUser | None):
        """Call after login/logout/registration. Rebuilds the sidebar for the new role."""
        self.user = user or VISITOR
        self._views.clear()  # drop cached views so nothing shows the previous user's data
        while self.stack.count():
            w = self.stack.widget(0)
            self.stack.removeWidget(w)
            w.deleteLater()

        self.user_label.setText(f"{self.user.username}\n{ROLE_NAMES[self.user.role]}")
        self._build_nav()
        self.show_screen("home")

    def show_screen(self, key: str):
        """Switch to a screen by key (see SCREENS). Also highlights it in the sidebar."""
        view = self._get_view(key)
        self.stack.setCurrentWidget(view)

        # keep the sidebar highlight in sync, without re-triggering navigation
        self.nav.blockSignals(True)
        self.nav.setCurrentRow(-1)
        for i in range(self.nav.count()):
            if self.nav.item(i).data(Qt.UserRole) == key:
                self.nav.setCurrentRow(i)
        self.nav.blockSignals(False)

        if hasattr(view, "on_show"):
            view.on_show(self.user)

    def show_book(self, book_id):
        """Open the book detail screen (used by Home and Browse). UC-07, UC-08."""
        view = self._get_view("book")
        self.stack.setCurrentWidget(view)
        self.nav.blockSignals(True)
        self.nav.setCurrentRow(-1)
        self.nav.blockSignals(False)
        if hasattr(view, "load_book"):
            view.load_book(book_id, self.user)

    # ---------- internals ----------

    def _build_nav(self):
        self.nav.blockSignals(True)
        self.nav.clear()
        for key in NAV_BY_ROLE[self.user.role]:
            label = LOGOUT_LABEL if key == "logout" else SCREENS[key].label
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, key)
            self.nav.addItem(item)
        self.nav.blockSignals(False)

    def _get_view(self, key: str) -> QWidget:
        if key not in self._views:
            view = self._create_view(SCREENS[key])
            self._views[key] = view
            self.stack.addWidget(view)
        return self._views[key]

    def _create_view(self, screen: Screen) -> QWidget:
        try:
            module = import_module(screen.module)
            view_class = getattr(module, screen.class_name)
            return view_class(self)
        except (ModuleNotFoundError, AttributeError) as exc:
            # file or class not written yet; only hide the error if it is *this* module missing
            missing_other = isinstance(exc, ModuleNotFoundError) and exc.name != screen.module
            return PlaceholderView(self, screen, str(exc) if missing_other else "")
        except Exception as exc:  # the real view crashed: show why instead of killing the app
            return PlaceholderView(self, screen, f"{type(exc).__name__}: {exc}")

    def _on_nav_changed(self, current, _previous):
        if current is None:
            return
        key = current.data(Qt.UserRole)
        if key == "logout":
            self.set_user(VISITOR)
        else:
            self.show_screen(key)

    def _on_dev_role_changed(self, _index):
        role = self.dev_roles.currentData()
        self.set_user(CurrentUser(username=f"dev_{role.lower()}", role=role) if role != VS else VISITOR)
