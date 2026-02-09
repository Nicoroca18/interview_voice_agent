# Voice Interview Agent

Conversational AI system that conducts short structured interviews using text or voice, extracts key information, and generates a final assessment report. The project demonstrates LLM integration, conversation management, structured data extraction, and production-style error handling.

---

## Overview

The agent simulates an interviewer who gathers candidate information through a multi-turn conversation. It maintains context, validates inputs, extracts structured fields, and stores results in JSON format.

Core capabilities:

- Natural multi-turn conversation
- Structured data extraction
- Conversation memory
- Automatic summary generation
- Sentiment and recommendation assessment
- Voice interface (Whisper STT + TTS)
- Persistent structured storage
- Graceful error handling

---

## Architecture

```
User (Voice/Text)
   → VoiceHandler (STT/TTS)
   → InterviewAgent (state manager)
   → LLM prompt engine
   → Structured extraction
   → JSON storage
```

Key components:

**InterviewAgent**
- Controls conversation lifecycle
- Maintains context
- Extracts structured data
- Generates final report

**VoiceHandler**
- Speech transcription
- Text-to-speech output
- Audio validation and recovery

**Storage Layer**
- Conversation logs
- Extracted metadata
- Final summary

The modular design supports scalability, multi-language extension, RAG integration, and API deployment.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add API keys to `.env`:

```
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=optional
```


Create data folders:

```
mkdir -p data/conversations data/summaries data/knowledge_base
```

Run test:

```
python main_voice.py --test-only
```

Run main version:

```
python main_voice.py
```

---

## Usage

Text interview:

```bash
python main.py
```

Voice interview:

```bash
python main_voice.py
```

Options:

```bash
--tts gtts | openai
--whisper-model tiny/base/small/medium/large
--test-only
--no-audio-test
```

---

## Output

Each interview generates:

- Full transcript
- Extracted structured fields
- Candidate metadata
- Summary + assessment

Stored in:

```
data/conversations/
data/summaries/
```

JSON format enables analytics and downstream automation.

---

## Conversation Flow

1. Agent introduces interview
2. Collects structured information
3. Validates and clarifies inputs
4. Maintains context across turns
5. Detects exit intent
6. Generates final report
7. Saves structured output

Handles silence, invalid input, interruptions, and early termination safely.

---

## Design Decisions

- Prompt-controlled structured extraction
- State-driven conversation management
- Defensive error handling
- Modular architecture
- JSON-first persistence
- Voice abstraction layer

The system is built to evolve into a web service, API endpoint, or multi-user conversational platform.

---

## Potential Improvements

- RAG knowledge base integration
- Web interface
- Real-time streaming transcription
- Multi-language auto-detection
- Candidate scoring models
- Analytics dashboard
- Containerized deployment
- Multi-session concurrency

---

## Sample Output

```json
{
  "candidate_name": "Jane Doe",
  "years_of_experience": 4,
  "skills": ["Python", "Cloud"],
  "recommendation": "Recommended",
  "sentiment": "Positive"
}
```

---

## Requirements

- Python 3.9+
- Microphone & speakers (voice mode)
- Internet connection
- Configured API keys
