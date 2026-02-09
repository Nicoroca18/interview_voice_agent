"""
Conversation management module for maintaining context and flow.
"""
from typing import List, Dict, Optional, Any
from datetime import datetime


class ConversationManager:
    """Manage conversation context and history."""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize the conversation manager.
        
        Args:
            max_history: Maximum number of conversation turns to keep in context
        """
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []
        self.metadata: Dict[str, Any] = {
            "start_time": datetime.now().isoformat(),
            "current_stage": "introduction",
            "information_collected": {},
            "questions_asked": []
        }
    
    def add_turn(self, role: str, content: str) -> None:
        """
        Add a conversation turn to history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
        """
        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        self.conversation_history.append(turn)
        
        # Keep only recent history to avoid context overflow
        if len(self.conversation_history) > self.max_history * 2:
            # Keep first 2 turns (introduction) + recent turns
            self.conversation_history = (
                self.conversation_history[:2] +
                self.conversation_history[-(self.max_history * 2 - 2):]
            )
    
    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """
        Get conversation history formatted for LLM API.
        
        Returns:
            List of messages without timestamps
        """
        return [
            {"role": turn["role"], "content": turn["content"]}
            for turn in self.conversation_history
        ]
    
    def get_full_history(self) -> List[Dict[str, str]]:
        """
        Get complete conversation history with timestamps.
        
        Returns:
            Full conversation history
        """
        return self.conversation_history.copy()
    
    def update_stage(self, stage: str) -> None:
        """
        Update the current interview stage.
        
        Args:
            stage: New stage name
        """
        self.metadata["current_stage"] = stage
        self.metadata["stage_changed_at"] = datetime.now().isoformat()
    
    def record_information(self, key: str, value: Any) -> None:
        """
        Record information collected during the interview.
        
        Args:
            key: Information key
            value: Information value
        """
        self.metadata["information_collected"][key] = {
            "value": value,
            "collected_at": datetime.now().isoformat()
        }
    
    def record_question(self, question: str) -> None:
        """
        Record a question that was asked.
        
        Args:
            question: The question asked
        """
        self.metadata["questions_asked"].append({
            "question": question,
            "asked_at": datetime.now().isoformat()
        })
    
    def get_missing_information(self, required_fields: List[str]) -> List[str]:
        """
        Get list of required information that hasn't been collected.
        
        Args:
            required_fields: List of required field names
            
        Returns:
            List of missing fields
        """
        collected = set(self.metadata["information_collected"].keys())
        required = set(required_fields)
        return list(required - collected)
    
    def get_context_summary(self) -> str:
        """
        Get a summary of the current conversation context.
        
        Returns:
            Context summary string
        """
        total_turns = len(self.conversation_history)
        current_stage = self.metadata.get("current_stage", "unknown")
        info_collected = len(self.metadata["information_collected"])
        questions_asked = len(self.metadata["questions_asked"])
        
        return f"""Conversation Context:
- Total turns: {total_turns}
- Current stage: {current_stage}
- Information items collected: {info_collected}
- Questions asked: {questions_asked}
- Started at: {self.metadata['start_time']}
"""
    
    def reset(self) -> None:
        """Reset the conversation state."""
        self.conversation_history.clear()
        self.metadata = {
            "start_time": datetime.now().isoformat(),
            "current_stage": "introduction",
            "information_collected": {},
            "questions_asked": []
        }
    
    def get_conversation_duration(self) -> str:
        """
        Get estimated conversation duration.
        
        Returns:
            Duration string
        """
        start_time = datetime.fromisoformat(self.metadata["start_time"])
        duration = datetime.now() - start_time
        
        minutes = int(duration.total_seconds() // 60)
        seconds = int(duration.total_seconds() % 60)
        
        return f"{minutes}m {seconds}s"
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get conversation metadata.
        
        Returns:
            Metadata dictionary
        """
        return self.metadata.copy()
