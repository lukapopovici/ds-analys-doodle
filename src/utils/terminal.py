import threading
from datetime import datetime
from typing import List, Dict, Optional


class Terminal:
    """A minimal singleton terminal logger.

    - DOES NOT render messages in Streamlit (per your request).
    - Has an `enabled` flag that controls whether calls to `log` are recorded.
    - Provide small helper to toggle `enabled` from a Streamlit UI (the helper
      does not render messages either — only toggles the state and shows meta info).
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, max_messages: int = 1000):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, max_messages: int = 1000):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.max_messages = max_messages
        self._messages: List[Dict] = []
        self._msg_lock = threading.Lock()
        self._enabled = False

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable logging."""
        self._enabled = bool(enabled)

    def enable(self) -> None:
        """Enable logging."""
        self.set_enabled(True)

    def disable(self) -> None:
        """Disable logging."""
        self.set_enabled(False)

    def is_enabled(self) -> bool:
        """Return True if the terminal is enabled and will accept logs."""
        return bool(self._enabled)

    def log(self, message: str, level: str = "INFO") -> bool:
        """Append a timestamped message to the terminal if enabled.

        Returns True if the message was recorded, False if the terminal is disabled.
        """
        if not self.is_enabled():
            return False

        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level.upper(),
            "message": str(message)
        }
        with self._msg_lock:
            self._messages.append(entry)
            if len(self._messages) > self.max_messages:
                self._messages = self._messages[-self.max_messages:]
        return True

    def clear(self) -> None:
        """Clear terminal history."""
        with self._msg_lock:
            self._messages = []

    def get_messages(self) -> List[Dict]:
        """Return a copy of logged messages."""
        with self._msg_lock:
            return list(self._messages)

    def last(self, n: int = 50) -> List[Dict]:
        """Return the last `n` messages (most recent last)."""
        with self._msg_lock:
            return list(self._messages[-n:])

    def message_count(self) -> int:
        """Return number of stored messages."""
        with self._msg_lock:
            return len(self._messages)


def terminal_toggle_ui(st, session_key: str = "terminal_enabled"):
    """Streamlit helper to toggle terminal enabled state from the UI.

    - Shows a single checkbox to turn the terminal on/off and a small status summary.
    - This helper NEVER displays the terminal message contents.

    Usage inside a Streamlit page:
        import streamlit as st
        from utils.terminal import terminal_toggle_ui

        terminal_toggle_ui(st)
    """
    try:
        import streamlit as st_mod
    except Exception:
        raise RuntimeError("Streamlit not available; cannot render terminal toggle UI")

    term = Terminal()

    if session_key not in st.session_state:
        st.session_state[session_key] = term.is_enabled()

    new_val = st.checkbox("Enable terminal logging", value=st.session_state[session_key])
    if new_val != st.session_state[session_key]:
        st.session_state[session_key] = new_val
        term.set_enabled(new_val)

    status = "ENABLED" if term.is_enabled() else "DISABLED"
    st.write(f"Terminal status: **{status}** — {term.message_count()} messages stored")

    if st.button("Clear terminal history"):
        term.clear()
        st.success("Terminal history cleared")


terminal = Terminal()
