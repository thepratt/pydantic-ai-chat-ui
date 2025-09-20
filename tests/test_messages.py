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
