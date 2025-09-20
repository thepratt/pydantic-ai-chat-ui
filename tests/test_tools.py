import pytest

from pydantic_ai_chat_ui.messages import DataPartState
from pydantic_ai_chat_ui.tools import get_tool_message


def test_get_tool_message_default_pending():
  msg = get_tool_message("search", DataPartState.PENDING, None)
  assert msg == "Calling `search`"


def test_get_tool_message_default_success():
  msg = get_tool_message("lookup", DataPartState.SUCCESS, None)
  assert msg == "Called `lookup` successfully"


def test_get_tool_message_default_error():
  msg = get_tool_message("fetch", DataPartState.ERROR, None)
  assert msg == "Error while calling `fetch`"


def test_get_tool_message_string_override_applies_to_all_states():
  tool_messages = {"search": "Running search"}
  for state in (DataPartState.PENDING, DataPartState.SUCCESS, DataPartState.ERROR):
    assert get_tool_message("search", state, tool_messages) == "Running search"


def test_get_tool_message_dict_override_per_state():
  tool_messages = {
    "tool": {
      DataPartState.PENDING: "Starting",
      DataPartState.SUCCESS: "Done",
      DataPartState.ERROR: "Failed",
    }
  }
  assert get_tool_message("tool", DataPartState.PENDING, tool_messages) == "Starting"
  assert get_tool_message("tool", DataPartState.SUCCESS, tool_messages) == "Done"
  assert get_tool_message("tool", DataPartState.ERROR, tool_messages) == "Failed"


def test_get_tool_message_invalid_override_type_raises():
  tool_messages = {"x": 123}
  with pytest.raises(NotImplementedError):
    get_tool_message("x", DataPartState.PENDING, tool_messages)


def test_get_tool_message_missing_tool_uses_default():
  tool_messages = {"other": "irrelevant"}
  assert (
    get_tool_message("missing", DataPartState.PENDING, tool_messages)
    == "Calling `missing`"
  )
