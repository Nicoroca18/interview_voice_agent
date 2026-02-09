"""
LLM Client for interacting with Claude API.
"""
import anthropic
from typing import List, Dict, Optional
from src.config import ANTHROPIC_API_KEY, MODEL_NAME


class LLMClient:
    """Client for interacting with Claude API."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the LLM client.
        
        Args:
            api_key: Anthropic API key (defaults to config)
            model: Model name to use (defaults to config)
        """
        self.api_key = api_key or ANTHROPIC_API_KEY
        self.model = model or MODEL_NAME
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set in environment or passed to constructor")
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """
        Generate a response from Claude.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_prompt: Optional system prompt to guide the model
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
            
        Returns:
            Generated response text
        """
        try:
            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }
            
            if system_prompt:
                kwargs["system"] = system_prompt
            
            response = self.client.messages.create(**kwargs)
            
            # Extract text from response
            return response.content[0].text
            
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
    
    def generate_structured_output(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        max_tokens: int = 2048
    ) -> str:
        """
        Generate structured output (like JSON) from Claude.
        
        Args:
            messages: List of message dictionaries
            system_prompt: System prompt requesting structured format
            max_tokens: Maximum tokens in response
            
        Returns:
            Structured response text
        """
        return self.generate_response(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.3  # Lower temperature for more consistent structured output
        )
    
    async def generate_response_async(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """
        Async version of generate_response.
        
        Args:
            messages: List of message dictionaries
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        # For now, we'll use the sync version wrapped
        # In production, you'd use anthropic.AsyncAnthropic
        return self.generate_response(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
