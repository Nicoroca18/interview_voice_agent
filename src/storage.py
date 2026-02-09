"""
Storage module for persisting conversations and extracted data.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from src.config import CONVERSATIONS_DIR, SUMMARIES_DIR
from src.validators import InterviewData


class StorageManager:
    """Manage storage of interview data."""
    
    def __init__(
        self,
        conversations_dir: Path = CONVERSATIONS_DIR,
        summaries_dir: Path = SUMMARIES_DIR
    ):
        """
        Initialize the storage manager.
        
        Args:
            conversations_dir: Directory for storing conversations
            summaries_dir: Directory for storing summaries
        """
        self.conversations_dir = Path(conversations_dir)
        self.summaries_dir = Path(summaries_dir)
        
        # Ensure directories exist
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
    
    def save_conversation(
        self,
        interview_id: str,
        conversation: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save conversation to JSON file.
        
        Args:
            interview_id: Unique identifier for the interview
            conversation: List of conversation turns
            metadata: Optional metadata to include
            
        Returns:
            Path to saved file
        """
        data = {
            "interview_id": interview_id,
            "timestamp": datetime.now().isoformat(),
            "conversation": conversation,
            "metadata": metadata or {}
        }
        
        filepath = self.conversations_dir / f"{interview_id}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def save_interview_data(self, interview_data: InterviewData) -> Path:
        """
        Save complete interview data.
        
        Args:
            interview_data: InterviewData object to save
            
        Returns:
            Path to saved file
        """
        filepath = self.conversations_dir / f"{interview_data.interview_id}_complete.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(interview_data.to_dict(), f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def save_summary(self, interview_id: str, summary: str) -> Path:
        """
        Save interview summary to text file.
        
        Args:
            interview_id: Unique identifier for the interview
            summary: Summary text
            
        Returns:
            Path to saved file
        """
        filepath = self.summaries_dir / f"{interview_id}_summary.txt"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Interview Summary - {interview_id}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(summary)
        
        return filepath
    
    def load_conversation(self, interview_id: str) -> Optional[Dict[str, Any]]:
        """
        Load conversation from JSON file.
        
        Args:
            interview_id: Unique identifier for the interview
            
        Returns:
            Dictionary with conversation data or None if not found
        """
        filepath = self.conversations_dir / f"{interview_id}.json"
        
        if not filepath.exists():
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_interview_data(self, interview_id: str) -> Optional[InterviewData]:
        """
        Load complete interview data.
        
        Args:
            interview_id: Unique identifier for the interview
            
        Returns:
            InterviewData object or None if not found
        """
        filepath = self.conversations_dir / f"{interview_id}_complete.json"
        
        if not filepath.exists():
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return InterviewData(**data)
    
    def list_interviews(self) -> List[str]:
        """
        List all stored interview IDs.
        
        Returns:
            List of interview IDs
        """
        interviews = []
        for filepath in self.conversations_dir.glob("*.json"):
            if not filepath.stem.endswith("_complete"):
                interviews.append(filepath.stem)
        return sorted(interviews)
    
    def export_to_json(self, interview_id: str, output_path: Optional[Path] = None) -> Path:
        """
        Export interview data to a specific location.
        
        Args:
            interview_id: Unique identifier for the interview
            output_path: Optional custom output path
            
        Returns:
            Path to exported file
        """
        data = self.load_conversation(interview_id)
        
        if data is None:
            raise FileNotFoundError(f"Interview {interview_id} not found")
        
        if output_path is None:
            output_path = Path(f"{interview_id}_export.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored data.
        
        Returns:
            Dictionary with storage statistics
        """
        conversations = list(self.conversations_dir.glob("*.json"))
        summaries = list(self.summaries_dir.glob("*.txt"))
        
        total_size = sum(f.stat().st_size for f in conversations)
        total_size += sum(f.stat().st_size for f in summaries)
        
        return {
            "total_interviews": len([c for c in conversations if not c.stem.endswith("_complete")]),
            "total_files": len(conversations) + len(summaries),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "conversations_dir": str(self.conversations_dir),
            "summaries_dir": str(self.summaries_dir)
        }
