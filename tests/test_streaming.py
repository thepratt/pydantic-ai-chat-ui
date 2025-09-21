import json

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import Tool

from pydantic_ai_chat_ui.messages.full import (
  DataPartState,
  MessageRole,
  TextPart,
  UIMessage,
)
from pydantic_ai_chat_ui.messages.streamed import (
  CodeArtifactData,
  DocumentArtifactData,
  TextPartDelta,
)
from pydantic_ai_chat_ui.streaming import format_event, stream_results


def test_format_event_wraps_with_data_prefix():
  evt = TextPartDelta(id="m1", delta="hello")
  s = format_event(evt)
  assert s.startswith("data: ") and s.endswith("\n\n")
  payload = s[len("data: ") : -2]
  data = json.loads(payload)
  assert data["type"] == "text-delta" and data["delta"] == "hello"


# Exercise the error path of stream_results using a real Agent configured with
# the TestModel, but passing a UIMessage (which the agent doesn't accept),
# causing an exception that stream_results handles by emitting an error part.


@pytest.mark.asyncio
async def test_stream_results_text_only_with_test_model():
  agent = Agent(model=TestModel())
  ui = UIMessage(id="u1", role=MessageRole.USER, parts=[TextPart(id="p1", text="hi")])

  chunks = []
  async for c in stream_results(
    user_message=ui,
    agent=agent,
    deps=None,
    message_history=[],
  ):
    chunks.append(c)

  # Expect text-start, at least one text-delta, and text-end
  payloads = [json.loads(c[len("data: ") : -2]) for c in chunks]
  types = [p.get("type") for p in payloads]
  assert (
    "text-start" in types
    and "text-end" in types
    and any(t == "text-delta" for t in types)
  )


@pytest.mark.asyncio
async def test_stream_results_tool_events_with_test_model_and_custom_messages():
  def hello() -> str:
    return "hello"

  tool = Tool.from_schema(
    function=hello,
    name="hello",
    description="say hello",
    json_schema={"type": "object", "properties": {}, "required": []},
  )

  agent = Agent(model=TestModel(call_tools="all"), tools=[tool])
  ui = UIMessage(id="u1", role=MessageRole.USER, parts=[TextPart(id="p1", text="hi")])

  tool_messages = {
    "hello": {DataPartState.PENDING: "Start", DataPartState.SUCCESS: "Done"}
  }

  chunks = []
  async for c in stream_results(
    user_message=ui,
    agent=agent,
    deps=None,
    message_history=[],
    tool_messages=tool_messages,
  ):
    chunks.append(c)

  payloads = [json.loads(c[len("data: ") : -2]) for c in chunks]
  events = [p for p in payloads if p.get("type") == "data-event"]
  statuses = [e["data"]["status"] for e in events]
  assert DataPartState.PENDING in statuses and DataPartState.SUCCESS in statuses
  # Also ensure custom messages appeared at least once
  all_titles = [e["data"].get("title") for e in events]
  assert "Start" in all_titles and "Done" in all_titles


@pytest.mark.asyncio
async def test_stream_results_artifacts_with_output_type():
  # Code artifact
  agent_code = Agent(
    model=TestModel(
      custom_output_args={"file_name": "x.py", "code": "print()", "language": "python"}
    ),
    output_type=CodeArtifactData,
  )
  ui = UIMessage(id="u2", role=MessageRole.USER, parts=[TextPart(id="p2", text="hi")])
  chunks = []
  async for c in stream_results(
    user_message=ui, agent=agent_code, deps=None, message_history=[]
  ):
    chunks.append(c)
  payloads = [json.loads(c[len("data: ") : -2]) for c in chunks]
  arts = [p for p in payloads if p.get("type") == "data-artifact"]
  assert any(a["data"]["type"] == "code" for a in arts)

  # Document artifact
  agent_doc = Agent(
    model=TestModel(custom_output_args={"title": "T", "content": "C", "type": "md"}),
    output_type=DocumentArtifactData,
  )
  chunks = []
  async for c in stream_results(
    user_message=ui, agent=agent_doc, deps=None, message_history=[]
  ):
    chunks.append(c)
  payloads = [json.loads(c[len("data: ") : -2]) for c in chunks]
  arts = [p for p in payloads if p.get("type") == "data-artifact"]
  assert any(a["data"]["type"] == "document" for a in arts)


@pytest.mark.asyncio
async def test_stream_results_error_and_error_event_with_raising_tool():
  def bad() -> str:
    raise RuntimeError("boom")

  tool = Tool.from_schema(
    function=bad,
    name="bad",
    description="bad",
    json_schema={"type": "object", "properties": {}, "required": []},
  )

  agent = Agent(model=TestModel(call_tools="all"), tools=[tool])
  ui = UIMessage(id="u3", role=MessageRole.USER, parts=[TextPart(id="p3", text="hi")])

  chunks = []
  async for c in stream_results(
    user_message=ui,
    agent=agent,
    deps=None,
    message_history=[],
    tool_messages={"bad": {}},
  ):
    chunks.append(c)

  # Expect a pending event for bad, an error event for bad, and an error part
  payloads = [json.loads(c[len("data: ") : -2]) for c in chunks]
  events = [p for p in payloads if p.get("type") == "data-event"]
  statuses = [e["data"]["status"] for e in events]
  assert "pending" in statuses and "error" in statuses
  assert any(p.get("type") == "error" for p in payloads)


@pytest.mark.asyncio
async def test_stream_results_store_message_history_called():
  agent = Agent(model=TestModel())
  ui = UIMessage(id="u1", role=MessageRole.USER, parts=[TextPart(id="p1", text="hi")])
  stored = []

  async for _ in stream_results(
    user_message=ui,
    agent=agent,
    deps=None,
    message_history=[],
    store_message_history=stored.append,
  ):
    pass

  assert len(stored) >= 1
