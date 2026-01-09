import threading
import sys
import os
from datetime import datetime
from typing import List, Dict, Optional
import subprocess


class Terminal:
    """A terminal logger that can write to system terminal/console when enabled.

    - Maintains in-memory message store
    - When enabled, also writes to stderr (visible in console/terminal)
    - Optionally can spawn a separate terminal window (Linux-specific)
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
        
        # External terminal process (for separate window)
        self._external_terminal = None
        self._use_external_term = False
        self._external_term_lock = threading.Lock()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable logging."""
        old_state = self._enabled
        self._enabled = bool(enabled)
        
        # If turning on external terminal
        if enabled and self._use_external_term and not old_state:
            self._start_external_terminal()
        # If turning off external terminal
        elif not enabled and self._use_external_term and old_state:
            self._stop_external_terminal()

    def enable(self) -> None:
        """Enable logging."""
        self.set_enabled(True)

    def disable(self) -> None:
        """Disable logging."""
        self.set_enabled(False)

    def is_enabled(self) -> bool:
        """Return True if the terminal is enabled and will accept logs."""
        return bool(self._enabled)

    def set_use_external_terminal(self, use_external: bool) -> None:
        """Set whether to use a separate terminal window."""
        old_setting = self._use_external_term
        self._use_external_term = use_external
        
        # Handle transitions
        if use_external and self._enabled and not old_setting:
            self._start_external_terminal()
        elif not use_external and old_setting:
            self._stop_external_terminal()

    def _start_external_terminal(self) -> None:
        """Start a separate terminal window (Linux-specific)."""
        with self._external_term_lock:
            if self._external_terminal is not None:
                return
            
            try:
                # Try different Linux terminal emulators
                terminals = ['gnome-terminal', 'konsole', 'xterm', 'xfce4-terminal', 'alacritty']
                
                for term in terminals:
                    try:
                        # Check if terminal exists
                        subprocess.run(['which', term], check=True, capture_output=True)
                        
                        # Start terminal with tail command or just a shell
                        if term == 'gnome-terminal':
                            self._external_terminal = subprocess.Popen([
                                'gnome-terminal', '--', 'tail', '-f', '/dev/null'
                            ])
                        elif term == 'konsole':
                            self._external_terminal = subprocess.Popen([
                                'konsole', '-e', 'tail', '-f', '/dev/null'
                            ])
                        else:
                            self._external_terminal = subprocess.Popen([
                                term, '-e', 'tail', '-f', '/dev/null'
                            ])
                        
                        self.log(f"Started external terminal: {term}", "INFO")
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
                        
                if self._external_terminal is None:
                    self._write_to_stderr("WARNING: No suitable terminal emulator found for external window")
                    
            except Exception as e:
                self._write_to_stderr(f"ERROR starting external terminal: {e}")

    def _stop_external_terminal(self) -> None:
        """Stop the external terminal window."""
        with self._external_term_lock:
            if self._external_terminal is not None:
                try:
                    self._external_terminal.terminate()
                    self._external_terminal.wait(timeout=2)
                    self.log("External terminal stopped", "INFO")
                except subprocess.TimeoutExpired:
                    self._external_terminal.kill()
                    self.log("External terminal force stopped", "WARN")
                finally:
                    self._external_terminal = None

    def _write_to_stderr(self, formatted_msg: str) -> None:
        """Write message to stderr (visible in console)."""
        print(formatted_msg, file=sys.stderr, flush=True)

    def _write_to_external_terminal(self, formatted_msg: str) -> None:
        """Write message to external terminal if available."""
        with self._external_term_lock:
            if self._external_terminal is not None and self._external_terminal.poll() is None:
                try:
                    # For demonstration - in reality, you'd need a pipe or socket
                    # This is complex to implement robustly without a proper IPC
                    self._write_to_stderr("(External terminal output not implemented in this example)")
                except Exception:
                    pass

    def log(self, message: str, level: str = "INFO", 
            include_timestamp: bool = True, color: bool = True) -> bool:
        """Append a timestamped message and optionally write to system terminal.
        
        Args:
            message: The log message
            level: Log level (INFO, WARN, ERROR, DEBUG)
            include_timestamp: Whether to include timestamp in console output
            color: Whether to use ANSI colors in console output
        
        Returns:
            True if message was recorded, False if terminal is disabled
        """
        if not self.is_enabled():
            return False

        # Store in memory
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level.upper(),
            "message": str(message)
        }
        
        with self._msg_lock:
            self._messages.append(entry)
            if len(self._messages) > self.max_messages:
                self._messages = self._messages[-self.max_messages:]

        # Write to system terminal (stderr)
        level_upper = level.upper()
        
        # ANSI color codes
        colors = {
            'INFO': '\033[94m',     # Blue
            'WARN': '\033[93m',     # Yellow
            'ERROR': '\033[91m',    # Red
            'DEBUG': '\033[90m',    # Gray
            'RESET': '\033[0m'      # Reset
        }
        
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            prefix = f"[{timestamp}] "
        else:
            prefix = ""
            
        if color and level_upper in colors:
            color_start = colors[level_upper]
            color_end = colors['RESET']
            formatted_msg = f"{prefix}{color_start}[{level_upper}]{color_end} {message}"
        else:
            formatted_msg = f"{prefix}[{level_upper}] {message}"
        
        # Write to stderr (always visible in console where app is running)
        self._write_to_stderr(formatted_msg)
        
        # Optionally write to external terminal
        if self._use_external_term:
            self._write_to_external_terminal(formatted_msg)
        
        return True

    def info(self, message: str, **kwargs) -> bool:
        """Log an INFO message."""
        return self.log(message, "INFO", **kwargs)

    def warn(self, message: str, **kwargs) -> bool:
        """Log a WARN message."""
        return self.log(message, "WARN", **kwargs)

    def error(self, message: str, **kwargs) -> bool:
        """Log an ERROR message."""
        return self.log(message, "ERROR", **kwargs)

    def debug(self, message: str, **kwargs) -> bool:
        """Log a DEBUG message."""
        return self.log(message, "DEBUG", **kwargs)

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
    """Streamlit helper to toggle terminal enabled state and settings.
    
    - Shows checkbox to enable/disable terminal
    - Option to use external terminal window
    - Status summary
    
    Usage:
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

    # Main enable/disable toggle
    new_val = st.checkbox("Enable terminal logging", value=st.session_state[session_key])
    if new_val != st.session_state[session_key]:
        st.session_state[session_key] = new_val
        term.set_enabled(new_val)

    # External terminal option (only show if enabled)
    if term.is_enabled():
        use_external = st.checkbox("Show in separate terminal window", value=False)
        term.set_use_external_terminal(use_external)
        
        # Log level selector
        log_level = st.selectbox(
            "Default log level",
            ["INFO", "WARN", "ERROR", "DEBUG"],
            index=0
        )
        
        # Test buttons
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Test INFO"):
                term.info(f"Test INFO message from Streamlit UI")
        with col2:
            if st.button("Test WARN"):
                term.warn(f"Test WARN message from Streamlit UI")
        with col3:
            if st.button("Test ERROR"):
                term.error(f"Test ERROR message from Streamlit UI")
        with col4:
            if st.button("Test DEBUG"):
                term.debug(f"Test DEBUG message from Streamlit UI")

    status = "ENABLED" if term.is_enabled() else "DISABLED"
    st.write(f"Terminal status: **{status}** — {term.message_count()} messages stored")

    if st.button("Clear terminal history"):
        term.clear()
        st.success("Terminal history cleared")


# Singleton instance
terminal = Terminal()