"""
Data extraction module for extracting structured information from conversations.
"""
import json
from typing import Dict, List, Any, Optional
from src.llm_client import LLMClient
from src.validators import CandidateInfo, InterviewAssessment, validate_candidate_info, validate_assessment


class DataExtractor:
    """Extract structured data from interview conversations."""
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize the data extractor.
        
        Args:
            llm_client: LLM client for generating extractions
        """
        self.llm_client = llm_client
    
    def extract_candidate_info(self, conversation: List[Dict[str, str]]) -> CandidateInfo:
        """
        Extract candidate information from conversation.
        
        Args:
            conversation: List of conversation turns
            
        Returns:
            Validated CandidateInfo object
        """
        system_prompt = """You are a data extraction assistant. Extract candidate information from the interview conversation.

Return ONLY a valid JSON object with these fields (use null for missing information):
{
    "candidate_name": "string or null",
    "email": "string or null",
    "phone": "string or null",
    "years_of_experience": "integer or null",
    "technical_skills": ["list", "of", "skills"],
    "previous_companies": ["list", "of", "companies"],
    "current_position": "string or null",
    "availability": "string or null",
    "expected_salary_range": "string or null",
    "education": "string or null",
    "location": "string or null",
    "willing_to_relocate": "boolean or null",
    "preferred_work_mode": "string or null"
}

Extract only information that was explicitly mentioned in the conversation. Do not infer or assume."""
        
        # Create a summary of the conversation for extraction
        conversation_text = self._format_conversation(conversation)
        
        messages = [
            {
                "role": "user",
                "content": f"Extract candidate information from this conversation:\n\n{conversation_text}"
            }
        ]
        
        try:
            response = self.llm_client.generate_structured_output(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=2048
            )
            
            # Parse JSON response
            data = self._parse_json_response(response)
            
            # Validate and return
            return validate_candidate_info(data)
            
        except Exception as e:
            # Return empty CandidateInfo if extraction fails
            print(f"Warning: Failed to extract candidate info: {str(e)}")
            return CandidateInfo()
    
    def extract_assessment(self, conversation: List[Dict[str, str]]) -> InterviewAssessment:
        """
        Extract interview assessment from conversation.
        
        Args:
            conversation: List of conversation turns
            
        Returns:
            Validated InterviewAssessment object
        """
        system_prompt = """You are an interview assessment assistant. Analyze the conversation and provide an assessment.

Return ONLY a valid JSON object with these fields:
{
    "communication_score": "integer 1-10 or null",
    "technical_score": "integer 1-10 or null",
    "cultural_fit_score": "integer 1-10 or null",
    "overall_sentiment": "positive/neutral/negative or null",
    "confidence_level": "high/medium/low or null",
    "red_flags": ["list", "of", "concerns"],
    "strengths": ["list", "of", "strengths"],
    "areas_for_concern": ["list", "of", "areas"],
    "recommendation": "proceed/reject/needs_more_info or null",
    "notes": "string or null"
}

Base your assessment only on what was discussed in the conversation."""
        
        conversation_text = self._format_conversation(conversation)
        
        messages = [
            {
                "role": "user",
                "content": f"Assess this interview conversation:\n\n{conversation_text}"
            }
        ]
        
        try:
            response = self.llm_client.generate_structured_output(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=2048
            )
            
            data = self._parse_json_response(response)
            return validate_assessment(data)
            
        except Exception as e:
            print(f"Warning: Failed to extract assessment: {str(e)}")
            return InterviewAssessment()
    
    def generate_summary(self, conversation: List[Dict[str, str]]) -> str:
        """
        Generate a summary of the interview conversation.
        
        Args:
            conversation: List of conversation turns
            
        Returns:
            Summary text
        """
        system_prompt = """You are a professional interview summarizer. Create a concise but comprehensive summary of the interview conversation.

Include:
- Key information shared by the candidate
- Main topics discussed
- Notable strengths or concerns
- Overall impression

Keep it professional and objective."""
        
        conversation_text = self._format_conversation(conversation)
        
        messages = [
            {
                "role": "user",
                "content": f"Summarize this interview:\n\n{conversation_text}"
            }
        ]
        
        try:
            return self.llm_client.generate_response(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=1024,
                temperature=0.5
            )
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def _format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Format conversation for LLM processing."""
        formatted = []
        for turn in conversation:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            speaker = "Interviewer" if role == "assistant" else "Candidate"
            formatted.append(f"{speaker}: {content}")
        return "\n\n".join(formatted)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        response = response.strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {str(e)}\nResponse: {response}")
