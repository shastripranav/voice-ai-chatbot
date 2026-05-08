# Contributing to voice-ai-chatbot

MIT licensed, contributions welcome. Useful contributions: additional STT/LLM/TTS provider adapters, improved interruption handling, and example client integrations beyond the included demo UI.

## How to Contribute

1. Fork the repository on GitHub.
2. Create a topic branch off `main` (e.g. `feat/azure-stt-provider`).
3. Make your changes and run the test suite with mock providers.
4. Open a pull request describing the change.

## Development setup

```bash
pip install -e ".[dev]"
```

The default `.env.example` configures all three pipeline stages (STT, LLM, TTS) to mock providers, so you can iterate without any API keys. To test against real providers, copy `.env.example` to `.env` and uncomment the relevant lines.

## Code style

```bash
ruff check src/ api/ tests/
```

## Testing

Tests run against mock providers by default:

```bash
pytest -v
```

When adding a new provider, please add a corresponding fixture-based test that exercises the provider interface against canned responses. Don't add tests that hit live APIs in CI.

## Questions

Open an issue with the `question` label.
