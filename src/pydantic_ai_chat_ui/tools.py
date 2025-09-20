from pydantic_ai_chat_ui.messages import DataPartState

ToolMessages = dict[str, dict[DataPartState, str] | str]


def get_tool_message(
  tool_name: str, state: DataPartState, tool_messages: ToolMessages | None
) -> str:
  default_messages: dict[DataPartState, str] = {
    DataPartState.PENDING: f"Calling `{tool_name}`",
    DataPartState.SUCCESS: f"Called `{tool_name}` successfully",
    DataPartState.ERROR: f"Error while calling `{tool_name}`",
  }
  if tool_messages is None:
    return default_messages[state]

  overridden_message = tool_messages.get(tool_name, None)
  if overridden_message is None:
    return default_messages[state]

  if isinstance(overridden_message, str):
    return overridden_message
  elif isinstance(overridden_message, dict):
    return overridden_message[state]
  else:
    raise NotImplementedError("Unsupported tool_message override")
