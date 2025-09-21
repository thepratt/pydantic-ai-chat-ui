from pydantic import BaseModel

from pydantic_ai_chat_ui.messages.full import UIMessage


class ChatRequest[U](BaseModel):
  id: U | None = None
  messages: list[UIMessage]
