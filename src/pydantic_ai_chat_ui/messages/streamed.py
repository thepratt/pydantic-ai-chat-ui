import enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pydantic_ai_chat_ui.messages.shared import (
  Artifact,
  ArtifactType,
  CodeArtifactData,
  DocumentArtifactData,
  FileData,
  SourceData,
)
from pydantic_ai_chat_ui.tools import DataPartState


@enum.verify(enum.UNIQUE)
class StreamedPartType(enum.StrEnum):
  """Streaming part type constants supported by chat-ui"""

  TEXT_START = "text-start"
  TEXT_DELTA = "text-delta"
  TEXT_END = "text-end"
  FILE = "data-file"
  ARTIFACT = "data-artifact"
  EVENT = "data-event"
  SOURCES = "data-sources"
  SUGGESTIONS = "data-suggested_questions"
  ERROR = "error"


class StreamedMessagePartBase(BaseModel):
  type: str
  id: str

  def __str__(self):
    return self.model_dump_json(by_alias=True)


class TextPartStart(StreamedMessagePartBase):
  type: Literal[StreamedPartType.TEXT_START] = StreamedPartType.TEXT_START


class TextPartDelta(StreamedMessagePartBase):
  type: Literal[StreamedPartType.TEXT_DELTA] = StreamedPartType.TEXT_DELTA
  delta: str


class TextPartEnd(StreamedMessagePartBase):
  type: Literal[StreamedPartType.TEXT_END] = StreamedPartType.TEXT_END


class DataPart[T](StreamedMessagePartBase):
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
  type: Literal[StreamedPartType.FILE] = StreamedPartType.FILE


CodeArtifact = Artifact[CodeArtifactData, Literal[ArtifactType.CODE]]
DocumentArtifact = Artifact[DocumentArtifactData, Literal[ArtifactType.DOCUMENT]]


class ArtifactPart(DataPart[CodeArtifact | DocumentArtifact]):
  type: Literal[StreamedPartType.ARTIFACT] = StreamedPartType.ARTIFACT


class EventPart(DataPart[ChatEvent]):
  type: Literal[StreamedPartType.EVENT] = StreamedPartType.EVENT


class SourcesPart(DataPart[SourceData]):
  type: Literal[StreamedPartType.SOURCES] = StreamedPartType.SOURCES


class SuggestionPart(DataPart[SuggestedQuestionsData]):
  type: Literal[StreamedPartType.SUGGESTIONS] = StreamedPartType.SUGGESTIONS


class AnyPart(StreamedMessagePartBase):
  type: str
  data: Any | None = None
  id: str


class ErrorPart(BaseModel):
  model_config = ConfigDict(populate_by_name=True)
  type: Literal[StreamedPartType.ERROR] = StreamedPartType.ERROR
  error_text: str = Field(alias="errorText")

  def __str__(self):
    return self.model_dump_json(by_alias=True)
