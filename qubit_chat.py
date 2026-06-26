#!/usr/bin/env python3
"""Convenience launcher for the Qubit TUI chatbot.

Equivalent to ``python -m tui_chat.app``. Lets you run the chatbot directly::

    python qubit_chat.py
    python qubit_chat.py --backend neuroquantum-model
"""

from tui_chat.app import main

if __name__ == "__main__":
    main()
