"""
Data validators for extracted information.
"""
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class CandidateInfo(BaseModel):
    """Structured candidate information extracted from interview."""
    
    model_config = ConfigDict(extra='allow')
    
    candidate_name: Optional[str] = Field(None, description="Full name of the candidate")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    years_of_experience: Optional[int] = Field(None, description="Years of professional experience")
    technical_skills: List[str] = Field(default_factory=list, description="List of technical skills")
    previous_companies: List[str] = Field(default_factory=list, description="Previous employers")
    current_position: Optional[str] = Field(None, description="Current job title")
    availability: Optional[str] = Field(None, description="When they can start")
    expected_salary_range: Optional[str] = Field(None, description="Expected salary")
    education: Optional[str] = Field(None, description="Education background")
    location: Optional[str] = Field(None, description="Current location")
    willing_to_relocate: Optional[bool] = Field(None, description="Willing to relocate")
    preferred_work_mode: Optional[str] = Field(None, description="Remote/Hybrid/Onsite preference")
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        """Validate email format."""
        if v is None:
            return v
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, v):
            raise ValueError(f"Invalid email format: {v}")
        return v
    
    @field_validator('years_of_experience')
    @classmethod
    def validate_experience(cls, v: Optional[int]) -> Optional[int]:
        """Validate years of experience."""
        if v is not None and (v < 0 or v > 50):
            raise ValueError(f"Years of experience must be between 0 and 50, got {v}")
        return v
    
    @field_validator('phone')
    @classmethod
    def validate_phone(cls, v: Optional[str]) -> Optional[str]:
        """Basic phone validation."""
        if v is None:
            return v
        # Remove common separators and spaces
        cleaned = re.sub(r'[\s\-\(\)]', '', v)
        # Check if it contains only digits and optionally a + prefix
        if not re.match(r'^\+?\d{7,15}$', cleaned):
            raise ValueError(f"Invalid phone format: {v}")
        return v


class InterviewAssessment(BaseModel):
    """Assessment of the candidate during the interview."""
    
    model_config = ConfigDict(extra='allow')
    
    communication_score: Optional[int] = Field(None, ge=1, le=10, description="Communication skills (1-10)")
    technical_score: Optional[int] = Field(None, ge=1, le=10, description="Technical competency (1-10)")
    cultural_fit_score: Optional[int] = Field(None, ge=1, le=10, description="Cultural fit (1-10)")
    overall_sentiment: Optional[str] = Field(None, description="positive/neutral/negative")
    confidence_level: Optional[str] = Field(None, description="high/medium/low")
    red_flags: List[str] = Field(default_factory=list, description="Any concerning points")
    strengths: List[str] = Field(default_factory=list, description="Candidate strengths")
    areas_for_concern: List[str] = Field(default_factory=list, description="Areas that need clarification")
    recommendation: Optional[str] = Field(None, description="proceed/reject/needs_more_info")
    notes: Optional[str] = Field(None, description="Additional notes")


class InterviewData(BaseModel):
    """Complete interview data structure."""
    
    model_config = ConfigDict(extra='allow')
    
    interview_id: str
    timestamp: str
    candidate_info: CandidateInfo
    assessment: InterviewAssessment
    conversation_summary: str
    language_detected: Optional[str] = None
    total_turns: int = 0
    duration_estimate: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump()


def validate_candidate_info(data: Dict[str, Any]) -> CandidateInfo:
    """
    Validate and create CandidateInfo from dictionary.
    
    Args:
        data: Dictionary with candidate information
        
    Returns:
        Validated CandidateInfo object
        
    Raises:
        ValueError: If validation fails
    """
    try:
        return CandidateInfo(**data)
    except Exception as e:
        raise ValueError(f"Validation error: {str(e)}")


def validate_assessment(data: Dict[str, Any]) -> InterviewAssessment:
    """
    Validate and create InterviewAssessment from dictionary.
    
    Args:
        data: Dictionary with assessment information
        
    Returns:
        Validated InterviewAssessment object
        
    Raises:
        ValueError: If validation fails
    """
    try:
        return InterviewAssessment(**data)
    except Exception as e:
        raise ValueError(f"Validation error: {str(e)}")
