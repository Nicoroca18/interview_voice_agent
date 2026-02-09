# Interview Agent

Conversational AI system for conducting technical interviews. Supports voice and text interactions with automatic data extraction and candidate assessment.

## Quick Start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add ANTHROPIC_API_KEY to .env
python main_voice.py --test-only  # Test audio
python main_voice.py              # Start interview
```

## Features

- Voice interviews with Whisper STT and customizable TTS
- Structured data extraction to JSON
- RAG-based intelligent questioning
- Sentiment analysis and red flag detection
- Multi-language support (EN/ES/FR/DE)

## Usage

```bash
python main.py              # Text mode
python main_voice.py        # Voice mode
```

Voice options:
```bash
--tts gtts              # Free TTS
--tts openai            # Premium TTS (requires OPENAI_API_KEY)
--whisper-model small   # Better accuracy
--test-only             # Test audio setup only
```

## Output

- `data/conversations/` - Interview data (JSON)
- `data/summaries/` - Text summaries

## Requirements

Python 3.9+, microphone/speakers for voice mode

