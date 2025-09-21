import logging
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import datetime

from pydantic_ai import Agent
from pydantic_ai import messages as pydantic_ai_messages
from pydantic_ai.output import OutputDataT
from pydantic_ai.result import FinalResult
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_chat_ui.messages.full import (
  ArtifactType,
  DataPartState,
  UIMessage,
  from_ui_message,
)
from pydantic_ai_chat_ui.messages.streamed import (
  ArtifactPart,
  ChatEvent,
  CodeArtifact,
  CodeArtifactData,
  DocumentArtifact,
  DocumentArtifactData,
  ErrorPart,
  EventPart,
  StreamedMessagePartBase,
  TextPartDelta,
  TextPartEnd,
  TextPartStart,
)
from pydantic_ai_chat_ui.tools import ToolMessages, get_tool_message

logger = logging.getLogger(__name__)


DATA_PREFIX = "data"


def format_event(event: StreamedMessagePartBase) -> str:
  return f"{DATA_PREFIX}: {event}\n\n"


async def stream_results[D: AgentDepsT, R: OutputDataT](
  user_message: UIMessage,
  agent: Agent[D, R],
  deps: D,
  message_history: list[pydantic_ai_messages.ModelMessage],
  tool_messages: ToolMessages | None = None,
  store_message_history: Callable[[pydantic_ai_messages.ModelMessage], None]
  | None = None,
) -> AsyncIterator[str]:
  message_started = False
  message_streamed = False
  active_tool_ids = {}

  try:
    async with agent.iter(
      from_ui_message(user_message),
      deps=deps,
      message_history=message_history,
    ) as agent_run:
      # pydantic ai text messages don't have ids, it doesn't matter for the
      # consuming frontend, as long as they are consistent between data parts
      message_id = str(uuid.uuid4())

      async for node in agent_run:
        if Agent.is_user_prompt_node(node):
          continue

        elif Agent.is_model_request_node(node):
          async with node.stream(agent_run.ctx) as stream:
            async for event in stream:
              if isinstance(event, pydantic_ai_messages.PartStartEvent):
                match event.part:
                  case pydantic_ai_messages.TextPart():
                    message_started = True
                    yield format_event(TextPartStart(id=message_id))

                  case pydantic_ai_messages.ToolCallPart(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                  ):
                    active_tool_ids[tool_call_id] = tool_name
                    title = get_tool_message(
                      tool_name, DataPartState.PENDING, tool_messages
                    )
                    yield format_event(
                      EventPart(
                        id=tool_call_id,
                        data=ChatEvent(
                          title=title,
                          status=DataPartState.PENDING,
                        ),
                      )
                    )

              elif isinstance(
                event, pydantic_ai_messages.PartDeltaEvent
              ) and isinstance(event.delta, pydantic_ai_messages.TextPartDelta):
                message_streamed = True
                yield format_event(
                  TextPartDelta(
                    id=message_id, delta=event.delta.content_delta
                  )  # pragma: no cover
                )

        elif Agent.is_call_tools_node(node):
          async with node.stream(agent_run.ctx) as stream:
            async for event in stream:
              match event:
                case pydantic_ai_messages.FunctionToolCallEvent(part=part):
                  # Tool call starting - send pending status
                  active_tool_ids[part.tool_call_id] = part.tool_name
                  yield format_event(
                    EventPart(
                      id=part.tool_call_id,
                      data=ChatEvent(
                        title=get_tool_message(
                          part.tool_name, DataPartState.PENDING, tool_messages
                        ),
                        status=DataPartState.PENDING,
                      ),
                    )
                  )

                case pydantic_ai_messages.FunctionToolResultEvent(
                  tool_call_id=tool_call_id, result=result
                ):
                  # Tool call completed - send success status
                  del active_tool_ids[tool_call_id]
                  yield format_event(
                    EventPart(
                      id=tool_call_id,
                      data=ChatEvent(
                        title=get_tool_message(
                          result.tool_name, DataPartState.SUCCESS, tool_messages
                        ),
                        status=DataPartState.SUCCESS,
                      ),
                    )
                  )

        elif Agent.is_end_node(node) and isinstance(node.data, FinalResult):
          if node.data.tool_call_id:
            active_tool_ids.pop(node.data.tool_call_id, None)
            yield format_event(
              EventPart(
                id=node.data.tool_call_id,
                data=ChatEvent(
                  title=get_tool_message(
                    node.data.tool_name, DataPartState.SUCCESS, tool_messages
                  ),
                  status=DataPartState.SUCCESS,
                ),
              )
            )

          if not message_started:
            yield format_event(TextPartStart(id=message_id))

          if isinstance(node.data.output, str) and not message_streamed:
            yield format_event(
              TextPartDelta(
                id=message_id, delta=node.data.output.lstrip()
              )  # pragma: no cover
            )

          elif isinstance(node.data.output, CodeArtifactData):
            yield format_event(
              ArtifactPart(
                id=node.data.tool_call_id,
                data=CodeArtifact(
                  data=node.data.output,
                  created_at=int(datetime.now().timestamp()),
                  type=ArtifactType.CODE,
                ),
              )
            )

          elif isinstance(node.data.output, DocumentArtifactData):
            yield format_event(
              ArtifactPart(
                id=node.data.tool_call_id,
                data=DocumentArtifact(
                  data=node.data.output,
                  created_at=int(datetime.now().timestamp()),
                  type=ArtifactType.DOCUMENT,
                ),
              )
            )

          # End of message: close text and reset per-message state
          yield format_event(TextPartEnd(id=message_id))
          active_tool_ids.clear()
          message_streamed = False

      for message in agent_run.result.new_messages():
        if store_message_history is not None:
          store_message_history(message)

  except Exception as e:
    logger.error("Streaming failed", exc_info=True)

    # clear out active tool calls, otherwise they'll be stuck as pending
    for tool_id, tool_name in active_tool_ids.items():
      yield format_event(
        EventPart(
          id=tool_id,
          data=ChatEvent(
            title=get_tool_message(tool_name, DataPartState.ERROR, tool_messages),
            status=DataPartState.ERROR,
            # TODO: with optional args/data
          ),
        )
      )

    yield format_event(ErrorPart(error_text=str(e)))
