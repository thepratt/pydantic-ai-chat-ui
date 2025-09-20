from pydantic import BaseModel


class MessagePart(BaseModel):
  type: str
  text: str | None = None
  id: str | None = None
  data: dict | None = None


class ChatMessage(BaseModel):
  role: str | None = None
  content: str | None = None
  parts: list[MessagePart] | None = None


class ChatRequest[U](BaseModel):
  id: U | None = None
  messages: list[ChatMessage]
