import json

from pydantic_ai_chat_ui.messages.shared import ArtifactType
from pydantic_ai_chat_ui.messages.streamed import (
  ArtifactPart,
  CodeArtifact,
  CodeArtifactData,
  ErrorPart,
  TextPartDelta,
  TextPartEnd,
  TextPartStart,
)


def test_text_streamed_parts_str_is_json():
  start = TextPartStart(id="m1")
  delta = TextPartDelta(id="m1", delta="hi")
  end = TextPartEnd(id="m1")

  for part in (start, delta, end):
    s = str(part)
    assert s.startswith("{") and s.endswith("}")
    data = json.loads(s)
    assert data["id"] == "m1"
    assert "type" in data


def test_artifact_parts_json_contains_nested_data():
  ca = CodeArtifact(
    created_at=123,
    type=ArtifactType.CODE,
    data=CodeArtifactData(file_name="x.py", code="print()", language="python"),
  )
  ap = ArtifactPart(id="a1", data=ca)
  s = str(ap)
  data = json.loads(s)
  assert data["id"] == "a1"
  assert data["data"]["type"] == "code"
  assert data["data"]["data"]["file_name"] == "x.py"


def test_error_part_str_is_jsonlike():
  s = str(ErrorPart(error_text="boom"))
  # Depending on pydantic, aliases may or may not be used; just assert JSON-like
  assert "error" in s and "boom" in s
