"""
Main Interview Agent that orchestrates the interview process.
"""
import uuid
from datetime import datetime
from typing import Optional, Dict, List, Any
from src.llm_client import LLMClient
from src.conversation_manager import ConversationManager
from src.data_extractor import DataExtractor
from src.storage import StorageManager
from src.validators import InterviewData, CandidateInfo, InterviewAssessment
from src.rag_system import RAGSystem
from src.sentiment_analyzer import SentimentAnalyzer
from src.language_manager import LanguageManager
from src.config import (
    AGENT_NAME, COMPANY_NAME, POSITION, INTERVIEW_STAGES,
    REQUIRED_FIELDS, ENABLE_RAG, ENABLE_SENTIMENT, ENABLE_MULTILANG
)


class InterviewAgent:
    """Intelligent conversational agent for conducting interviews."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        enable_rag: bool = ENABLE_RAG,
        enable_sentiment: bool = ENABLE_SENTIMENT,
        enable_multilang: bool = ENABLE_MULTILANG
    ):
        """
        Initialize the interview agent.
        
        Args:
            api_key: Anthropic API key
            enable_rag: Enable RAG system
            enable_sentiment: Enable sentiment analysis
            enable_multilang: Enable multi-language support
        """
        # Core components
        self.llm_client = LLMClient(api_key=api_key)
        self.conversation_manager = ConversationManager()
        self.data_extractor = DataExtractor(self.llm_client)
        self.storage_manager = StorageManager()
        
        # Interview metadata
        self.interview_id = f"interview_{uuid.uuid4().hex[:8]}"
        self.current_stage = "introduction"
        self.candidate_language = "en"
        
        # Optional features
        self.rag_system = RAGSystem() if enable_rag else None
        self.sentiment_analyzer = SentimentAnalyzer() if enable_sentiment else None
        self.language_manager = LanguageManager() if enable_multilang else None
        
        # Interview state
        self.is_interview_complete = False
        self.asked_questions = []
        
    def start_interview(self) -> str:
        """
        Start a new interview session.
        
        Returns:
            Opening message from the agent
        """
        # Reset conversation state
        self.conversation_manager.reset()
        self.interview_id = f"interview_{uuid.uuid4().hex[:8]}"
        self.current_stage = "introduction"
        self.is_interview_complete = False
        self.asked_questions = []
        
        # Generate opening message
        opening_message = self._get_opening_message()
        
        # Add to conversation history
        self.conversation_manager.add_turn("assistant", opening_message)
        self.conversation_manager.update_stage("introduction")
        
        return opening_message
    
    def process_message(self, user_message: str) -> str:
        """
        Process a user message and generate a response.
        
        Args:
            user_message: Message from the candidate
            
        Returns:
            Agent's response
        """
        # Detect language if multilang is enabled
        if self.language_manager:
            detected_lang = self.language_manager.detect_language(user_message)
            self.candidate_language = detected_lang
            
            # Translate to English for processing if needed
            if detected_lang != 'en':
                user_message_english = self.language_manager.translate_to_english(
                    user_message, detected_lang
                )
            else:
                user_message_english = user_message
        else:
            user_message_english = user_message
        
        # Add user message to conversation
        self.conversation_manager.add_turn("user", user_message_english)
        
        # Analyze sentiment if enabled
        sentiment_info = None
        if self.sentiment_analyzer:
            sentiment_info = self.sentiment_analyzer.analyze_message(user_message_english)
            
            # Check for red flags
            if sentiment_info.get('is_frustrated'):
                # Adjust response to be more empathetic
                pass
        
        # Get relevant context from RAG if enabled
        rag_context = ""
        if self.rag_system:
            relevant_docs = self.rag_system.retrieve_relevant_context(
                user_message_english,
                n_results=2
            )
            if relevant_docs:
                rag_context = "\n".join([doc['content'] for doc in relevant_docs])
        
        # Generate response
        response = self._generate_response(
            user_message_english,
            rag_context=rag_context,
            sentiment_info=sentiment_info
        )
        
        # Translate response back to candidate's language if needed
        if self.language_manager and self.candidate_language != 'en':
            response_translated = self.language_manager.translate_from_english(
                response, self.candidate_language
            )
        else:
            response_translated = response
        
        # Add assistant message to conversation
        self.conversation_manager.add_turn("assistant", response)
        
        # Check if interview should end
        if self._should_end_interview():
            self.is_interview_complete = True
        
        return response_translated
    
    def end_interview(self) -> Dict[str, Any]:
        """
        End the interview and generate final report.
        
        Returns:
            Dictionary with interview results
        """
        conversation = self.conversation_manager.get_full_history()
        
        # Extract candidate information
        candidate_info = self.data_extractor.extract_candidate_info(
            self.conversation_manager.get_messages_for_llm()
        )
        
        # Generate assessment
        assessment = self.data_extractor.extract_assessment(
            self.conversation_manager.get_messages_for_llm()
        )
        
        # Add sentiment analysis to assessment if enabled
        if self.sentiment_analyzer:
            sentiment_analysis = self.sentiment_analyzer.analyze_conversation(
                conversation,
                role_filter='user'
            )
            assessment.overall_sentiment = sentiment_analysis['overall_sentiment']
            assessment.confidence_level = sentiment_analysis['overall_confidence']
            
            # Detect red flags
            red_flags = self.sentiment_analyzer.detect_red_flags(conversation)
            if red_flags:
                assessment.red_flags.extend(red_flags)
        
        # Generate summary
        summary = self.data_extractor.generate_summary(
            self.conversation_manager.get_messages_for_llm()
        )
        
        # Detect language
        detected_language = None
        if self.language_manager:
            detected_language = self.language_manager.get_dominant_language(conversation)
        
        # Create interview data object
        interview_data = InterviewData(
            interview_id=self.interview_id,
            timestamp=datetime.now().isoformat(),
            candidate_info=candidate_info,
            assessment=assessment,
            conversation_summary=summary,
            language_detected=detected_language,
            total_turns=len(conversation),
            duration_estimate=self.conversation_manager.get_conversation_duration()
        )
        
        # Save data
        self.storage_manager.save_interview_data(interview_data)
        self.storage_manager.save_conversation(
            self.interview_id,
            conversation,
            metadata=self.conversation_manager.get_metadata()
        )
        self.storage_manager.save_summary(self.interview_id, summary)
        
        return interview_data.to_dict()
    
    def _get_opening_message(self) -> str:
        """Generate opening message for the interview."""
        if self.language_manager:
            greeting = self.language_manager.get_prompt('greeting', self.candidate_language)
            return f"{greeting}\n\nMy name is {AGENT_NAME}, and we're hiring for a {POSITION} position at {COMPANY_NAME}. Let's get started!"
        else:
            return f"Hello! I'm {AGENT_NAME}, an AI recruiter. I'll be conducting your initial interview for the {POSITION} position at {COMPANY_NAME}. Let's get started!\n\nCould you please tell me your name?"
    
    def _generate_response(
        self,
        user_message: str,
        rag_context: str = "",
        sentiment_info: Optional[Dict] = None
    ) -> str:
        """Generate response using LLM."""
        system_prompt = self._build_system_prompt(rag_context, sentiment_info)
        
        messages = self.conversation_manager.get_messages_for_llm()
        
        try:
            response = self.llm_client.generate_response(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=512,
                temperature=0.7
            )
            
            # Record question if it's a question
            if '?' in response:
                self.conversation_manager.record_question(response)
                self.asked_questions.append(response)
            
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an error. Could you please repeat that? (Error: {str(e)})"
    
    def _build_system_prompt(
        self,
        rag_context: str = "",
        sentiment_info: Optional[Dict] = None
    ) -> str:
        """Build system prompt for the LLM."""
        prompt = f"""You are {AGENT_NAME}, an AI recruiter conducting a first-phase interview for a {POSITION} position at {COMPANY_NAME}.

Your responsibilities:
1. Conduct a professional, friendly interview
2. Gather information about the candidate's background, skills, and experience
3. Ask relevant follow-up questions based on their responses
4. Maintain a conversational and natural tone
5. Keep responses concise (2-3 sentences typically)

Information to collect:
- Name and contact information
- Years of experience
- Technical skills and expertise
- Previous companies and roles
- Availability and salary expectations
- Work preferences (remote/hybrid/onsite)

Current interview stage: {self.current_stage}

Guidelines:
- Be empathetic and encouraging
- Ask one question at a time
- Listen carefully and ask relevant follow-ups
- Don't be too formal or robotic
- If the candidate seems frustrated, be more supportive
- Keep the interview moving but don't rush
"""
        
        if rag_context:
            prompt += f"\n\nRelevant knowledge base context:\n{rag_context}\n"
        
        if sentiment_info:
            if sentiment_info.get('is_frustrated'):
                prompt += "\nNote: The candidate may be feeling frustrated. Be extra supportive and clear in your response.\n"
            
            if sentiment_info.get('confidence_level') == 'low':
                prompt += "\nNote: The candidate seems uncertain. Provide encouragement and clarify your questions if needed.\n"
        
        # Add context about missing information
        missing_info = self.conversation_manager.get_missing_information(REQUIRED_FIELDS)
        if missing_info:
            prompt += f"\nStill need to collect: {', '.join(missing_info)}\n"
        
        # Add context if interview should end soon
        total_turns = len(self.conversation_manager.get_full_history())
        if total_turns >= 15 and not missing_info:
            prompt += "\nNote: You have collected sufficient information. After answering the current question, prepare to close the interview politely. Include a closing statement like 'Thank you for your time today. This concludes our interview.'\n"
        elif total_turns >= 25:
            prompt += "\nNote: The interview has been quite long. Consider wrapping up the conversation with a polite closing statement.\n"
        
        return prompt
    
    def _should_end_interview(self) -> bool:
        """Determine if the interview should end."""
        # Check if we have enough conversation turns
        total_turns = len(self.conversation_manager.get_full_history())
        
        if total_turns < 10:
            return False
        
        # Check if we have collected essential information
        missing_info = self.conversation_manager.get_missing_information(
            ["candidate_name", "years_of_experience"]
        )
        
        # End if we have basic info and enough conversation
        if not missing_info and total_turns >= 15:
            return True
        
        # End if conversation is very long
        if total_turns >= 30:
            return True
        
        return False
    
    def get_interview_status(self) -> Dict[str, Any]:
        """Get current interview status."""
        return {
            "interview_id": self.interview_id,
            "current_stage": self.current_stage,
            "total_turns": len(self.conversation_manager.get_full_history()),
            "is_complete": self.is_interview_complete,
            "duration": self.conversation_manager.get_conversation_duration(),
            "language": self.candidate_language,
            "questions_asked": len(self.asked_questions)
        }
