# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Architecture Overview

This is a Python library that provides UI message types and streaming utilities for integrating pydantic-ai agents with chat-ui applications. The library bridges between pydantic-ai's message format and the structured message format expected by chat-ui frontends.

### Core Components

- **Message System**: Two parallel message type systems
  - `messages.py`: Full UI message objects returned to chat-ui
  - `streamed_messages.py`: Delta/streaming message parts for real-time updates
- **Streaming Engine**: `streaming.py` contains the main `stream_results()` function that processes pydantic-ai agent responses and converts them to chat-ui compatible streaming format
- **Request Types**: `requests.py` defines the chat request structure
- **Tool Integration**: `tools.py` provides utilities for customizing tool call messages in the UI

### Message Flow Architecture

1. **Input**: `ChatRequest` with list of `UIMessage` objects
2. **Processing**: `stream_results()` runs pydantic-ai agent and converts events
3. **Output**: Server-sent events formatted for chat-ui consumption

The library handles conversion between pydantic-ai's native message types and chat-ui's expected format, including:
- Text streaming (start/delta/end events)
- Tool call status updates (pending/success/error)
- Artifact rendering (code/document types)
- Error handling with proper UI feedback

## Development Commands

### Setup
```bash
# Initialize development environment (uses devbox)
devbox shell
```

### Dependencies
```bash
# Install/sync dependencies
uv sync

# Add new dependency
uv add <package>

# Add development dependency
uv add --group dev <package>
```

### Code Quality
```bash
# Format code
uv run ruff format

# Lint code
uv run ruff check

# Lint with auto-fix
uv run ruff check --fix

# Run pre-commit hooks manually
pre-commit run --all-files
```

### Package Management
```bash
# Build package
uv build

# Install package in development mode
uv pip install -e .
```

## Development Environment

- **Python Version**: 3.13.3 (specified in .python-version and devbox.json)
- **Package Manager**: uv (configured for build backend)
- **Environment Manager**: devbox with automatic venv setup
- **Code Quality**: ruff for formatting and linting
- **Pre-commit**: Configured with hooks for trailing whitespace, JSON formatting, ruff checks

## Key Implementation Notes

### Message Type Duality
The library maintains two parallel message type hierarchies because chat-ui requires different data structures for:
- Complete messages (when loading conversation history)
- Streaming deltas (during real-time conversation)

### Tool Call Integration
Tool calls are converted to `EventPart` messages with status tracking. The `tools.py` module allows customization of tool messages displayed in the UI through the `ToolMessages` type.

### Error Handling
The streaming function includes comprehensive error handling that ensures tool calls don't get stuck in "pending" state and provides meaningful error messages to the UI.

### Chat-UI Compatibility
All data parts must have a "data-" prefix in their type field to be recognized by chat-ui. The library enforces this through type definitions and validation.
