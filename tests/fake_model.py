"""Scripted fake Model API for tests — no network.

Provides ScriptedMuse: maps turn index to tool call, plus SAM outputs from recorded cache.
"""

from __future__ import annotations

import json
from typing import Callable
from unittest.mock import MagicMock


def tool_call(name: str, arguments: dict, call_id: str | None = None) -> dict:
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": call_id or f"call_{name}",
                    "function": {"name": name, "arguments": json.dumps(arguments)}
                }]
            }
        }]
    }


class ScriptedMuse:
    """Scripted planner for labvision.agent._call_muse.

    planner: function(messages, turn) -> (tool_name, args) or None for plain text
    sam_outputs: dict image_hash+prompt -> Segmentation fake
    """
    def __init__(self, planner: Callable[[list, int], tuple | None], sam_outputs: dict | None = None):
        self.planner = planner
        self.sam_outputs = sam_outputs or {}
        self.turn = 0
        self.calls = []

    def __call__(self, messages, tools=None, use_cache=True):
        self.turn += 1
        self.calls.append(messages)
        action = self.planner(messages, self.turn)
        if action is None:
            return {"choices": [{"message": {"role": "assistant", "content": "All done. Good."}}]}
        name, args = action
        return tool_call(name, args)

    def as_side_effect(self):
        def _se(*a, **kw):
            return self(*a, **kw)
        return _se
