"""
These message definitions are for "full" objects being returned to chat-ui, not
deltas or streamed data. For those see pydantic_ai_chat_ui.streamed_messages.
"""

import enum
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import messages as pydantic_ai_messages

from pydantic_ai_chat_ui.messages.shared import (
  Artifact,
  ArtifactType,
  CodeArtifactData,
  DocumentArtifactData,
  FileData,
  SourceData,
)
from pydantic_ai_chat_ui.tools import DataPartState, ToolMessages, get_tool_message


@enum.verify(enum.UNIQUE)
class MessageRole(enum.StrEnum):
  SYSTEM = "system"
  USER = "user"
  ASSISTANT = "assistant"


@enum.verify(enum.UNIQUE)
class PartType(enum.StrEnum):
  """Part type constants supported by chat-ui"""

  TEXT = "text"
  FILE = "data-file"
  ARTIFACT = "data-artifact"
  EVENT = "data-event"
  SOURCES = "data-sources"
  SUGGESTIONS = "data-suggested_questions"


class MessagePartBase(BaseModel):
  type: str
  id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class TextPart(MessagePartBase):
  type: Literal[PartType.TEXT] = PartType.TEXT
  text: str


class DataPart[T](MessagePartBase):
  """
  Generic data part for structured content.
  Type must have 'data-' prefix for chat-ui compatibility.
  """

  type: str  # Must start with 'data-'
  data: T
  id: str


class ChatEvent(BaseModel):
  title: str
  status: DataPartState


class SuggestedQuestionsData(BaseModel):
  questions: list[str]


class FilePart(DataPart[FileData]):
  type: Literal[PartType.FILE] = PartType.FILE


CodeArtifact = Artifact[CodeArtifactData, Literal[ArtifactType.CODE]]
DocumentArtifact = Artifact[DocumentArtifactData, Literal[ArtifactType.DOCUMENT]]


class ArtifactPart(DataPart[CodeArtifact | DocumentArtifact]):
  type: Literal[PartType.ARTIFACT] = PartType.ARTIFACT


class EventPart(DataPart[ChatEvent]):
  type: Literal[PartType.EVENT] = PartType.EVENT


class SourcesPart(DataPart[SourceData]):
  type: Literal[PartType.SOURCES] = PartType.SOURCES


class SuggestionPart(DataPart[SuggestedQuestionsData]):
  type: Literal[PartType.SUGGESTIONS] = PartType.SUGGESTIONS


class AnyPart(MessagePartBase):
  type: str
  data: Any | None = None
  id: str


UIMessagePart = (
  TextPart
  | FilePart
  | ArtifactPart
  | EventPart
  | SourcesPart
  | SuggestionPart
  | AnyPart
)


class UIMessage(BaseModel):
  id: str
  role: MessageRole
  parts: list[UIMessagePart]


def from_ui_message(message: UIMessage) -> pydantic_ai_messages.UserContent | None:
  """
  This is limited to supporting user-driven submissions. Longer sets of model
  messages should be loaded from your store of choice and included as message
  history, or a pydantic ai agnostic memory augmentation like mem0.

  Only text formats are handled, no multi-modal support.
  """
  if message.role != MessageRole.USER:
    return None

  texts: list[str] = []
  for part in message.parts:
    if isinstance(part, TextPart):
      texts.append(part.text)

  if not texts:
    return ""

  if len(texts) == 1:
    return texts[0]

  return "\n\n".join(texts)


def from_pydantic_ai_message(
  message: pydantic_ai_messages.ModelMessage,
  tool_messages: ToolMessages | None = None,
) -> UIMessage:
  message_parts = []

  if isinstance(message, pydantic_ai_messages.ModelRequest):
    role = MessageRole.USER

    for part in message.parts:
      match part:
        case pydantic_ai_messages.UserPromptPart(content=content):
          message_parts.append(TextPart(id=str(uuid.uuid4()), text=content))

        case pydantic_ai_messages.ToolReturnPart(
          tool_call_id=tool_call_id, tool_name=tool_name
        ):
          # Convert tool results to data-event parts
          event = EventPart(
            id=tool_call_id,
            data=ChatEvent(
              title=get_tool_message(tool_name, DataPartState.SUCCESS, tool_messages),
              status=DataPartState.SUCCESS,
            ),
          )
          message_parts.append(event)

        case pydantic_ai_messages.RetryPromptPart(tool_name=tool_name) if tool_name:
          event = EventPart(
            id=str(uuid.uuid4()),
            data=ChatEvent(
              title=get_tool_message(tool_name, DataPartState.ERROR, tool_messages),
              status=DataPartState.ERROR,
            ),
          )
          message_parts.append(event)

  elif isinstance(message, pydantic_ai_messages.ModelResponse):
    role = MessageRole.ASSISTANT

    for part in message.parts:
      if isinstance(part, pydantic_ai_messages.TextPart):
        message_parts.append(TextPart(id=str(uuid.uuid4()), text=part.content))

      elif isinstance(part, pydantic_ai_messages.ToolCallPart):
        # Convert tool calls to data-event parts
        event = EventPart(
          id=part.tool_call_id,
          data=ChatEvent(
            title=get_tool_message(
              part.tool_name, DataPartState.PENDING, tool_messages
            ),
            status=DataPartState.PENDING,
          ),
        )
        message_parts.append(event)

      elif isinstance(part, pydantic_ai_messages.ToolReturnPart):
        # Convert tool results to data-event parts
        event = EventPart(
          id=part.tool_call_id,
          data=ChatEvent(
            title=get_tool_message(
              part.tool_name, DataPartState.SUCCESS, tool_messages
            ),
            status=DataPartState.SUCCESS,
          ),
        )
        message_parts.append(event)

  return UIMessage(
    id=str(uuid.uuid4()),
    role=role,
    parts=message_parts or [TextPart(id=str(uuid.uuid4()), text="")],
  )
