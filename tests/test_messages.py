from pydantic_ai import messages as pa

import pydantic_ai_chat_ui.messages as ui_messages
from pydantic_ai_chat_ui.messages import (
  DataPartState,
  EventPart,
  MessageRole,
  TextPart,
  UIMessage,
)


def test_from_model_request_user_prompt_to_ui_text():
  msg = pa.ModelRequest(parts=[pa.UserPromptPart(content="hello")])
  ui = ui_messages.from_pydantic_ai_message(msg)
  assert isinstance(ui, UIMessage)
  assert ui.role == MessageRole.USER
  assert len(ui.parts) == 1
  assert isinstance(ui.parts[0], TextPart)
  assert ui.parts[0].text == "hello"


def test_from_model_response_text_to_ui_text():
  msg = pa.ModelResponse(parts=[pa.TextPart(content="world")])
  ui = ui_messages.from_pydantic_ai_message(msg)
  assert ui.role == MessageRole.ASSISTANT
  assert len(ui.parts) == 1
  assert isinstance(ui.parts[0], TextPart)
  assert ui.parts[0].text == "world"


def test_tool_call_and_return_to_events_with_status_and_titles():
  tool_messages = {
    "tool": {
      DataPartState.PENDING: "Calling it",
      DataPartState.SUCCESS: "Did it",
      DataPartState.ERROR: "Failed it",
    }
  }

  resp = pa.ModelResponse(parts=[pa.ToolCallPart(tool_call_id="tc1", tool_name="tool")])
  ui1 = ui_messages.from_pydantic_ai_message(resp, tool_messages)
  evt1 = next(p for p in ui1.parts if isinstance(p, EventPart))
  assert evt1.data.data["status"] == DataPartState.PENDING
  assert evt1.data.data["title"] == "Calling it"

  resp2 = pa.ModelResponse(
    parts=[
      pa.ToolReturnPart(tool_call_id="tc1", tool_name="tool", content="ok"),
    ]
  )
  ui2 = ui_messages.from_pydantic_ai_message(resp2, tool_messages)
  evt2 = next(p for p in ui2.parts if isinstance(p, EventPart))
  assert evt2.data.data["status"] == DataPartState.SUCCESS
  assert evt2.data.data["title"] == "Did it"


def test_tool_result_on_request_and_retry_error_event():
  tool_messages = {"tool": {DataPartState.ERROR: "Oops"}}

  req = pa.ModelRequest(
    parts=[pa.ToolReturnPart(tool_call_id="tcx", tool_name="tool", content="ok")]
  )
  ui_req = ui_messages.from_pydantic_ai_message(req, tool_messages)
  evt = next(p for p in ui_req.parts if isinstance(p, EventPart))
  assert evt.data.data["status"] == DataPartState.SUCCESS

  req_retry = pa.ModelRequest(
    parts=[pa.RetryPromptPart(tool_name="tool", content="retry")]
  )
  ui_retry = ui_messages.from_pydantic_ai_message(req_retry, tool_messages)
  evt_r = next(p for p in ui_retry.parts if isinstance(p, EventPart))
  assert evt_r.data.data["status"] == DataPartState.ERROR


def test_empty_parts_fallback_to_empty_text():
  resp = pa.ModelResponse(parts=[])
  ui = ui_messages.from_pydantic_ai_message(resp)
  assert len(ui.parts) == 1
  assert isinstance(ui.parts[0], TextPart)
  assert ui.parts[0].text == ""


def test_from_ui_message_single_text_to_model_request():
  user_ui = ui_messages.UIMessage(
    id="u",
    role=ui_messages.MessageRole.USER,
    parts=[ui_messages.TextPart(id="p", text="hello")],
  )
  req = ui_messages.from_ui_message(user_ui)
  assert isinstance(req, pa.ModelRequest)
  assert len(req.parts) == 1
  assert isinstance(req.parts[0], pa.UserPromptPart)
  assert req.parts[0].content == "hello"


def test_from_ui_message_multiple_text_parts():
  user_ui = ui_messages.UIMessage(
    id="u",
    role=ui_messages.MessageRole.USER,
    parts=[
      ui_messages.TextPart(id="p1", text="hello"),
      ui_messages.TextPart(id="p2", text="world"),
    ],
  )
  req = ui_messages.from_ui_message(user_ui)
  assert len(req.parts) == 2
  assert [p.content for p in req.parts] == ["hello", "world"]


def test_from_ui_message_ignores_non_text_and_falls_back_empty():
  user_ui = ui_messages.UIMessage(
    id="u",
    role=ui_messages.MessageRole.USER,
    parts=[
      ui_messages.EventPart(id="e", data=ui_messages.ChatEvent(data={"x": 1})),
    ],
  )
  req = ui_messages.from_ui_message(user_ui)
  assert len(req.parts) == 1
  assert isinstance(req.parts[0], pa.UserPromptPart)
  assert req.parts[0].content == ""


def test_from_ui_message_mixed_parts_order_preserved():
  # Interleave text with non-text parts; only text should be used, in order
  user_ui = ui_messages.UIMessage(
    id="u",
    role=ui_messages.MessageRole.USER,
    parts=[
      ui_messages.TextPart(id="t1", text="one"),
      ui_messages.EventPart(id="e1", data=ui_messages.ChatEvent(data={"x": 1})),
      ui_messages.TextPart(id="t2", text="two"),
    ],
  )
  req = ui_messages.from_ui_message(user_ui)
  assert [p.content for p in req.parts] == ["one", "two"]


def test_from_ui_message_empty_parts_fallback_only_once():
  # No parts -> exactly one empty prompt part
  user_ui = ui_messages.UIMessage(id="u", role=ui_messages.MessageRole.USER, parts=[])
  req = ui_messages.from_ui_message(user_ui)
  assert len(req.parts) == 1 and req.parts[0].content == ""


def test_from_ui_message_text_empty_string_kept_no_extra():
  # An explicit empty text becomes one empty prompt (not two)
  user_ui = ui_messages.UIMessage(
    id="u",
    role=ui_messages.MessageRole.USER,
    parts=[ui_messages.TextPart(id="t", text="")],
  )
  req = ui_messages.from_ui_message(user_ui)
  assert len(req.parts) == 1 and req.parts[0].content == ""


def test_from_ui_message_ignores_file_and_sources():
  # File and sources parts are ignored; only text is converted
  file_part = ui_messages.FilePart(
    id="f1",
    data=ui_messages.FileData(name="a.txt", url="http://x", type="text/plain", size=10),
  )
  sources_part = ui_messages.SourcesPart(
    id="s1",
    data=ui_messages.SourceData(sources=[{"title": "doc", "url": "http://d"}]),
  )
  user_ui = ui_messages.UIMessage(
    id="u",
    role=ui_messages.MessageRole.USER,
    parts=[file_part, ui_messages.TextPart(id="t", text="use this"), sources_part],
  )
  req = ui_messages.from_ui_message(user_ui)
  assert len(req.parts) == 1 and req.parts[0].content == "use this"


def test_from_ui_message_non_user_returns_none():
  assistant_ui = ui_messages.UIMessage(
    id="a",
    role=ui_messages.MessageRole.ASSISTANT,
    parts=[ui_messages.TextPart(id="p", text="hi")],
  )
  assert ui_messages.from_ui_message(assistant_ui) is None
