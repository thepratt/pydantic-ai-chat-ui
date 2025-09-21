# pydantic-ai-chat-ui

[![PyPI version](https://img.shields.io/pypi/v/pydantic-ai-chat-ui.svg)](https://pypi.org/project/pydantic-ai-chat-ui/)
[![Python versions](https://img.shields.io/pypi/pyversions/pydantic-ai-chat-ui.svg)](https://pypi.org/project/pydantic-ai-chat-ui/)
[![License](https://img.shields.io/pypi/l/pydantic-ai-chat-ui.svg)](LICENSE)
[![CI](https://github.com/thepratt/pydantic-ai-chat-ui/actions/workflows/test.yml/badge.svg)](https://github.com/thepratt/pydantic-ai-chat-ui/actions/workflows/test.yml)

A lightweight adapter that allows [Pydantic AI](https://ai.pydantic.dev/) agents to work with Vercel's Data Stream format via `useChat`, and more specifically the [LlamaIndex Chat UI](https://github.com/run-llama/chat-ui) flavour. The adpater supports:

- streaming (SSE) real-time messages from Pydantic AI agents to your frontend, including intermediate tool calls
- conversion of historical Pydantic AI messages, should you store them and want to return a full message history

Check out:

- Vercel AI SDK (TypeScript): https://github.com/vercel/ai (>=5)
- LlamaIndex Chat UI (React): https://github.com/run-llama/chat-ui

## Installation

```bash
uv add pydantic-ai-chat-ui
```

```bash
poetry add pydantic-ai-chat-ui
```

```bash
pip install pydantic-ai-chat-ui
```

## Quickstart Example

Not all logic is implemented, but hopefully it's enough of a guide to point you in the right direction.

### Route (FastAPI)

```python path=/src/your_app/app/routers/chat.py
import logging
import uuid

import logfire
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic_ai_chat_ui import ChatRequest, stream_results
from pydantic_ai_chat_ui.messages import from_pydantic_ai_message
from pydantic_ai_chat_ui.tools import DataPartState
from voyageai.client_async import AsyncClient as AsyncVoyageClient

from your_app.agents import your_agent
from your_app.agents.operative_report_state import (
  OperativeReportDeps,
  OperativeReportState,
)
from your_app.app.security import UserIdDep
from your_app.config import settings
from your_app.database import SessionDep
from your_app.threads import (
  create_or_get_thread,
  get_thread,
  store_message,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat")
async def agent_chat(
  chat_request: ChatRequest, user_id: UserIdDep, db_session: SessionDep
) -> StreamingResponse:
  thread = create_or_get_thread(chat_request.id, user_id, db_session=db_session)

  deps = Deps(
    thread_id=thread.id,
    voyage_client=AsyncVoyageClient(settings.VOYAGE_API_KEY),
    db_session=db_session,
  )

  return StreamingResponse(
    stream_results(
      chat_request.messages[0],
      your_agent.agent,
      deps,
      message_history=thread.messages,
      tool_messages={
        "validate_data": {
          DataPartState.PENDING: "Validating data...",
          DataPartState.SUCCESS: "Data validated.",
          DataPartState.ERROR: "Encountered an error during validation.",
        },
        "identify_stuff": {
          DataPartState.PENDING: "Identifying and analysing stuff...",
          DataPartState.SUCCESS: "Stuff identified.",
          DataPartState.ERROR: "Encountered an error during identification.",
        },
      },
      store_message_history=lambda message: store_message(
        thread.id, message, db_session=db_session
      ),
    ),
    media_type="text/event-stream",
    headers={
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
      # this is an important header for having types picked up
      "X-Vercel-AI-UI-Message-Stream": "v1",
    },
  )
```

### Frontend

```tsx path=/frontend/src/(app)/chat.tsx
import { useChat } from "@ai-sdk/react";
import {
  ChatCanvas,
  ChatInput,
  ChatMessage,
  ChatMessages,
  ChatSection,
  useChatUI,
} from "@llamaindex/chat-ui";
import { createFileRoute } from "@tanstack/react-router";
import { DefaultChatTransport, type UIMessage } from "ai";
import { toast } from "sonner";
import { v4 as uuid4 } from "uuid";

import "@llamaindex/chat-ui/styles/editor.css";
import "@llamaindex/chat-ui/styles/markdown.css";
import "@llamaindex/chat-ui/styles/pdf.css";
import "@mdxeditor/editor/style.css";

export const Route = createFileRoute("/(app)/chat")({
  component: ChatRoute,
});

function ChatRoute() {
  const initialMessages: UIMessage[] = [];
  const { messages, status, sendMessage, stop, regenerate, setMessages } =
    useChat({
      transport: new DefaultChatTransport({
        api: `${import.meta.env.VITE_BASE_API_URL}/chat`,
      }),
      generateId: () => uuid4(),
      messages: initialMessages,
      onError: (err) => toast.error(err.message),
    });

  const CustomChatMessages = () => {
    const { messages } = useChatUI();

    return (
      <>
        {messages.map((message, idx) => (
          <ChatMessage
            key={`message-${message.id}`}
            message={message}
            isLast={idx === messages.length - 1}
          >
            <ChatMessage.Avatar />
            <ChatMessage.Content>
              <ChatMessage.Part.Markdown />
              <ChatMessage.Part.Artifact />
              <ChatMessage.Part.Event />
              <ChatMessage.Part.Suggestion />
            </ChatMessage.Content>
            <ChatMessage.Actions />
          </ChatMessage>
        ))}
      </>
    );
  };

  return (
    <ChatSection
      handler={{
        messages,
        status,
        sendMessage,
        stop,
        regenerate,
        setMessages,
      }}
      className="h-full flex-row gap-4 p-0 flex md:p-5"
    >
      <div className="mx-auto flex h-full min-w-0 max-w-full flex-1 flex-col gap-4">
        <ChatMessages>
          <ChatMessages.List>
            <CustomChatMessages />
          </ChatMessages.List>
          <ChatMessages.Empty />
          <ChatMessages.Loading />
        </ChatMessages>

        <ChatInput>
          <ChatInput.Form>
            <ChatInput.Field className="max-h-32" />
            <ChatInput.Submit />
          </ChatInput.Form>
        </ChatInput>
      </div>
      <ChatCanvas className="w-full md:w-2/3" />
    </ChatSection>
  );
}
```
