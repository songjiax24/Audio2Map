"""Token-cursor failures. These do not reject a chart by themselves."""

from __future__ import annotations


class IllegalTokenError(ValueError):
    """``token`` is not legal in the current cursor."""

    def __init__(self, token: str) -> None:
        self.token = token
        super().__init__(token)
