#!/usr/bin/env python3
"""Convenience launcher for the Qubit AI TUI chatbot.

Equivalent to ``python -m qubit_ai_cli.app``. Lets you run the chatbot directly::

    python qubit_ai_cli.py
    python qubit_ai_cli.py --backend neuroquantum-model
"""

from qubit_ai_cli.app import main

if __name__ == "__main__":
    main()
