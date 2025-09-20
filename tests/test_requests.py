import uuid

from pydantic_ai_chat_ui.messages import MessageRole, TextPart, UIMessage
from pydantic_ai_chat_ui.requests import ChatRequest


def test_chat_request_instantiation():
  ui_msg = UIMessage(
    id=str(uuid.uuid4()),
    role=MessageRole.USER,
    parts=[TextPart(id=str(uuid.uuid4()), text="hello")],
  )
  req = ChatRequest(id="abc", messages=[ui_msg])
  assert req.id == "abc"
  assert len(req.messages) == 1
  assert req.messages[0].role == MessageRole.USER
