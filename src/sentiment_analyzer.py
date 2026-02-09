"""
Sentiment analysis module for detecting candidate emotions and confidence.
"""
from typing import Dict, List, Optional
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    """Analyze sentiment and emotional tone in conversations."""
    
    def __init__(self):
        """Initialize sentiment analyzers."""
        self.vader = SentimentIntensityAnalyzer()
        
        # Keywords for detecting frustration
        self.frustration_keywords = [
            'frustrated', 'annoyed', 'confused', 'difficult', 'hard',
            'unclear', 'don\'t understand', 'not sure', 'struggling'
        ]
        
        # Keywords for detecting confidence
        self.confidence_keywords = {
            'high': ['confident', 'sure', 'definitely', 'absolutely', 'experienced', 'expert', 'proficient'],
            'low': ['maybe', 'perhaps', 'not sure', 'uncertain', 'think so', 'probably', 'guess']
        }
    
    def analyze_message(self, text: str) -> Dict[str, any]:
        """
        Analyze sentiment of a single message.
        
        Args:
            text: Message text
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # VADER sentiment analysis
        vader_scores = self.vader.polarity_scores(text)
        
        # TextBlob sentiment analysis
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment
        
        # Determine overall sentiment
        compound = vader_scores['compound']
        if compound >= 0.05:
            overall = 'positive'
        elif compound <= -0.05:
            overall = 'negative'
        else:
            overall = 'neutral'
        
        # Check for frustration
        is_frustrated = any(keyword in text.lower() for keyword in self.frustration_keywords)
        
        # Assess confidence level
        confidence = self._assess_confidence(text)
        
        return {
            'overall_sentiment': overall,
            'compound_score': compound,
            'positive_score': vader_scores['pos'],
            'neutral_score': vader_scores['neu'],
            'negative_score': vader_scores['neg'],
            'polarity': textblob_sentiment.polarity,
            'subjectivity': textblob_sentiment.subjectivity,
            'is_frustrated': is_frustrated,
            'confidence_level': confidence
        }
    
    def analyze_conversation(
        self,
        conversation: List[Dict[str, str]],
        role_filter: Optional[str] = 'user'
    ) -> Dict[str, any]:
        """
        Analyze sentiment across entire conversation.
        
        Args:
            conversation: List of conversation turns
            role_filter: Analyze only messages from this role ('user' or 'assistant')
            
        Returns:
            Aggregated sentiment analysis
        """
        # Filter messages by role if specified
        if role_filter:
            messages = [turn['content'] for turn in conversation if turn['role'] == role_filter]
        else:
            messages = [turn['content'] for turn in conversation]
        
        if not messages:
            return self._empty_analysis()
        
        # Analyze each message
        analyses = [self.analyze_message(msg) for msg in messages]
        
        # Aggregate results
        avg_compound = sum(a['compound_score'] for a in analyses) / len(analyses)
        avg_polarity = sum(a['polarity'] for a in analyses) / len(analyses)
        avg_subjectivity = sum(a['subjectivity'] for a in analyses) / len(analyses)
        
        # Count sentiment distribution
        sentiment_counts = {
            'positive': sum(1 for a in analyses if a['overall_sentiment'] == 'positive'),
            'neutral': sum(1 for a in analyses if a['overall_sentiment'] == 'neutral'),
            'negative': sum(1 for a in analyses if a['overall_sentiment'] == 'negative')
        }
        
        # Check if frustration was detected
        frustration_detected = any(a['is_frustrated'] for a in analyses)
        frustration_count = sum(1 for a in analyses if a['is_frustrated'])
        
        # Determine overall confidence trend
        confidence_levels = [a['confidence_level'] for a in analyses if a['confidence_level']]
        overall_confidence = self._aggregate_confidence(confidence_levels)
        
        # Determine overall sentiment
        if avg_compound >= 0.05:
            overall_sentiment = 'positive'
        elif avg_compound <= -0.05:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'overall_sentiment': overall_sentiment,
            'average_compound_score': round(avg_compound, 3),
            'average_polarity': round(avg_polarity, 3),
            'average_subjectivity': round(avg_subjectivity, 3),
            'sentiment_distribution': sentiment_counts,
            'frustration_detected': frustration_detected,
            'frustration_instances': frustration_count,
            'overall_confidence': overall_confidence,
            'total_messages_analyzed': len(messages),
            'sentiment_trajectory': self._calculate_trajectory(analyses)
        }
    
    def detect_red_flags(self, conversation: List[Dict[str, str]]) -> List[str]:
        """
        Detect potential red flags in conversation.
        
        Args:
            conversation: List of conversation turns
            
        Returns:
            List of detected red flags
        """
        red_flags = []
        
        user_messages = [turn['content'] for turn in conversation if turn['role'] == 'user']
        
        # Analyze overall sentiment
        analysis = self.analyze_conversation(conversation, role_filter='user')
        
        # Check for persistent negative sentiment
        if analysis['average_compound_score'] < -0.2:
            red_flags.append("Persistent negative sentiment detected")
        
        # Check for frustration
        if analysis['frustration_detected'] and analysis['frustration_instances'] > 2:
            red_flags.append("Multiple instances of frustration")
        
        # Check for very low confidence
        if analysis['overall_confidence'] == 'low':
            red_flags.append("Low confidence in responses")
        
        # Check for evasiveness (high number of uncertain terms)
        uncertain_count = sum(
            1 for msg in user_messages
            if any(term in msg.lower() for term in ['maybe', 'not sure', 'i guess', 'i think'])
        )
        if uncertain_count > len(user_messages) * 0.5:
            red_flags.append("Frequent use of uncertain language")
        
        return red_flags
    
    def _assess_confidence(self, text: str) -> Optional[str]:
        """Assess confidence level from text."""
        text_lower = text.lower()
        
        high_count = sum(1 for keyword in self.confidence_keywords['high'] if keyword in text_lower)
        low_count = sum(1 for keyword in self.confidence_keywords['low'] if keyword in text_lower)
        
        if high_count > low_count:
            return 'high'
        elif low_count > high_count:
            return 'low'
        elif high_count == low_count and high_count > 0:
            return 'medium'
        return None
    
    def _aggregate_confidence(self, confidence_levels: List[str]) -> str:
        """Aggregate confidence levels across multiple messages."""
        if not confidence_levels:
            return 'medium'
        
        counts = {
            'high': confidence_levels.count('high'),
            'medium': confidence_levels.count('medium'),
            'low': confidence_levels.count('low')
        }
        
        return max(counts, key=counts.get)
    
    def _calculate_trajectory(self, analyses: List[Dict]) -> str:
        """Calculate sentiment trajectory (improving, declining, stable)."""
        if len(analyses) < 3:
            return 'insufficient_data'
        
        # Compare first third vs last third
        third = len(analyses) // 3
        early_avg = sum(a['compound_score'] for a in analyses[:third]) / third
        late_avg = sum(a['compound_score'] for a in analyses[-third:]) / third
        
        diff = late_avg - early_avg
        
        if diff > 0.1:
            return 'improving'
        elif diff < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    def _empty_analysis(self) -> Dict[str, any]:
        """Return empty analysis when no messages."""
        return {
            'overall_sentiment': 'neutral',
            'average_compound_score': 0.0,
            'average_polarity': 0.0,
            'average_subjectivity': 0.0,
            'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
            'frustration_detected': False,
            'frustration_instances': 0,
            'overall_confidence': 'medium',
            'total_messages_analyzed': 0,
            'sentiment_trajectory': 'insufficient_data'
        }
