"""
Configuration module for the Interview Agent.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration 
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "claude-sonnet-4-20250514")

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" 
CONVERSATIONS_DIR = DATA_DIR / "conversations"
SUMMARIES_DIR = DATA_DIR / "summaries"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"

# Create directories if they don't exist
for directory in [CONVERSATIONS_DIR, SUMMARIES_DIR, KNOWLEDGE_BASE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
 
# Agent Configuration
AGENT_NAME = os.getenv("AGENT_NAME", "Julia")
COMPANY_NAME = os.getenv("COMPANY_NAME", "Orbio") 
POSITION = os.getenv("POSITION", "Software Engineer")

# Feature Flags
ENABLE_RAG = os.getenv("ENABLE_RAG", "true").lower() == "true"
ENABLE_SENTIMENT = os.getenv("ENABLE_SENTIMENT", "true").lower() == "true"
ENABLE_MULTILANG = os.getenv("ENABLE_MULTILANG", "true").lower() == "true"

# Language Settings
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")
SUPPORTED_LANGUAGES = os.getenv("SUPPORTED_LANGUAGES", "en,es,fr,de").split(",")

# Interview Questions (will be loaded from knowledge base)
INTERVIEW_STAGES = [
    "introduction",
    "background",
    "technical_skills",
    "experience",
    "behavioral",
    "closing"
]

# Required Information to Extract
REQUIRED_FIELDS = [
    "candidate_name",
    "years_of_experience",
    "technical_skills",
    "previous_companies",
    "availability",
    "expected_salary_range"
]
