import enum
from typing import Any

from pydantic import BaseModel


@enum.verify(enum.UNIQUE)
class ArtifactType(enum.StrEnum):
  CODE = "code"
  DOCUMENT = "document"


class FileData(BaseModel):
  name: str
  url: str
  type: str
  size: int


class Artifact[T, K: str](BaseModel):
  type: K
  data: T
  created_at: int  # timestamp


class CodeArtifactData(BaseModel):
  file_name: str
  code: str
  language: str


class DocumentArtifactData(BaseModel):
  title: str
  content: str
  type: str
  sources: list[dict[str, str]] | None = None


class SourceData(BaseModel):
  sources: list[dict[str, Any]]
