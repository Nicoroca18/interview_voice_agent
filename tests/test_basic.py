import pytest
from src.conversation_manager import ConversationManager
from src.validators import CandidateInfo
from src.sentiment_analyzer import SentimentAnalyzer

def test_conversation_manager():
    manager = ConversationManager()
    manager.add_turn('user', 'Hello')
    assert len(manager.get_full_history()) == 1

def test_candidate_validation():
    data = {
        'candidate_name': 'John Doe',
        'email': 'john@example.com',
        'years_of_experience': 5
    }
    candidate = CandidateInfo(**data)
    assert candidate.candidate_name == 'John Doe'

def test_sentiment_analysis():
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze_message("I'm excited about this opportunity!")
    assert result['overall_sentiment'] == 'positive'

def test_negative_sentiment():
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze_message("This is frustrating.")
    assert result['is_frustrated'] == True
