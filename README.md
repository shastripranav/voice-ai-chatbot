# Voice AI Chatbot Starter Kit

A modular voice conversation pipeline that chains Speech-to-Text, LLM processing, and Text-to-Speech into an end-to-end voice AI experience. Each component (STT, LLM, TTS) can be swapped independently via configuration.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│              Streamlit Demo UI                        │
│  [Record Button] → [Conversation] → [Audio Playback] │
└──────────────────┬───────────────────────────────────┘
                   │ Audio (wav/webm)
                   ▼
┌──────────────────────────────────────────────────────┐
│              FastAPI Backend                          │
│                                                      │
│  POST /api/chat/voice    ← upload audio, get audio   │
│  POST /api/chat/text     ← send text, get text       │
│  WS   /api/chat/stream   ← streaming voice chat      │
│  GET  /api/voices         ← list available TTS voices │
│  GET  /api/health         ← health check             │
└──────────────────┬───────────────────────────────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│   STT    │ │   LLM    │ │   TTS    │
│ Provider │ │ Provider │ │ Provider │
│          │ │          │ │          │
│ • Whisper│ │ • OpenAI │ │ • Eleven │
│ • Mock   │ │ • Claude │ │   Labs   │
│          │ │ • Ollama │ │ • Mock   │
│          │ │ • Mock   │ │          │
└──────────┘ └──────────┘ └──────────┘
```

## Quick Start

```bash
# Clone and install
cd voice-ai-chatbot
pip install -e ".[dev]"

# Copy env file and add your API keys (optional — mocks work without keys)
cp .env.example .env

# Start the API server
uvicorn api.main:app --reload

# In another terminal, launch the Streamlit demo
streamlit run app.py
```

### Run with Mock Providers (No API Keys)

```bash
STT_PROVIDER=mock LLM_PROVIDER=mock TTS_PROVIDER=mock uvicorn api.main:app --reload
```

## API Key Setup

| Provider | Env Variable | Get a Key |
|----------|-------------|-----------|
| OpenAI (Whisper + GPT) | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/api-keys) |
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) |
| ElevenLabs (TTS) | `ELEVENLABS_API_KEY` | [elevenlabs.io](https://elevenlabs.io/) |
| Ollama (local) | — | [ollama.com](https://ollama.com/) (no key needed) |

## API Reference

### POST `/api/chat/text`

Send a text message and get a text response.

```bash
curl -X POST http://localhost:8000/api/chat/text \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'
```

Response:
```json
{"response": "I'm doing well, thanks for asking!", "session_id": "abc123..."}
```

### POST `/api/chat/voice`

Upload audio and receive an audio response (full STT → LLM → TTS pipeline).

```bash
curl -X POST http://localhost:8000/api/chat/voice \
  -F "audio=@recording.wav" \
  -F "session_id=" \
  --output response.wav
```

### GET `/api/voices`

List available TTS voices (ElevenLabs provider only).

### GET `/api/health`

Returns provider status and configuration.

### WS `/api/chat/stream`

WebSocket endpoint for streaming voice conversation. Protocol:

```json
// Client sends:
{"type": "text", "data": "Hello"}
{"type": "audio", "data": "<base64>", "format": "wav"}

// Server responds:
{"type": "transcript", "data": "..."}
{"type": "response_text", "data": "..."}
{"type": "audio_chunk", "data": "<base64>"}
{"type": "done"}
```

## Configuration

All settings are configured via environment variables or a `.env` file.

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_PROVIDER` | `whisper` | STT provider: `whisper`, `mock` |
| `LLM_PROVIDER` | `openai` | LLM provider: `openai`, `anthropic`, `ollama`, `mock` |
| `TTS_PROVIDER` | `elevenlabs` | TTS provider: `elevenlabs`, `mock` |
| `OPENAI_API_KEY` | — | OpenAI API key (for Whisper + GPT) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (for Claude) |
| `ELEVENLABS_API_KEY` | — | ElevenLabs API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model to use |
| `ANTHROPIC_MODEL` | `claude-3-haiku-20240307` | Anthropic model |
| `OLLAMA_MODEL` | `llama3.1` | Ollama model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `ELEVENLABS_VOICE_ID` | `21m00Tcm4TlvDq8ikWAM` | ElevenLabs voice (Rachel) |
| `SYSTEM_PROMPT` | `You are a helpful AI assistant...` | System prompt for the LLM |
| `MAX_CONVERSATION_TURNS` | `20` | Max turns before trimming history |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests (uses mock providers, no API keys needed)
pytest

# Run with coverage
pytest --cov=voice_chatbot --cov=api

# Lint
ruff check src/ api/ tests/

# Auto-fix lint issues
ruff check --fix src/ api/ tests/
```

## Project Structure

```
src/voice_chatbot/
├── pipeline.py          # Main voice conversation pipeline orchestrator
├── conversation.py      # Conversation memory management
├── config.py            # Configuration (pydantic-settings)
├── utils.py             # Audio format helpers
├── stt/                 # Speech-to-Text providers
│   ├── base.py          #   Abstract interface
│   ├── whisper.py       #   OpenAI Whisper
│   └── mock.py          #   Mock (testing)
├── llm/                 # LLM providers
│   ├── base.py          #   Abstract interface
│   ├── openai_llm.py    #   OpenAI GPT
│   ├── anthropic_llm.py #   Anthropic Claude
│   ├── ollama_llm.py    #   Local Ollama
│   └── mock.py          #   Mock (testing)
└── tts/                 # Text-to-Speech providers
    ├── base.py          #   Abstract interface
    ├── elevenlabs.py    #   ElevenLabs
    └── mock.py          #   Mock (testing)

api/                     # FastAPI backend
├── main.py              #   App setup + middleware
├── routes.py            #   REST endpoints
└── ws.py                #   WebSocket streaming

app.py                   # Streamlit demo UI
```

## License

MIT
