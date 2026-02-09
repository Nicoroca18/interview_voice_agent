"""
Interview Agent - An intelligent conversational AI for conducting first-phase interviews.
"""

__version__ = "1.0.0"
__author__ = "Nicolas Roca"

from src.agent import InterviewAgent
from src.config import AGENT_NAME, COMPANY_NAME, POSITION

__all__ = ["InterviewAgent", "AGENT_NAME", "COMPANY_NAME", "POSITION"]
