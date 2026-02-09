"""
RAG (Retrieval-Augmented Generation) module for knowledge base integration.
"""
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from src.config import KNOWLEDGE_BASE_DIR


class RAGSystem:
    """RAG system for retrieving relevant information from knowledge base."""
    
    def __init__(
        self,
        knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR,
        collection_name: str = "interview_knowledge"
    ):
        """
        Initialize the RAG system.
        
        Args:
            knowledge_base_dir: Directory containing knowledge base files
            collection_name: Name of the ChromaDB collection
        """
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.collection_name = collection_name
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            is_persistent=False
        ))
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Interview knowledge base"}
        )
        
        # Load knowledge base if collection is empty
        if self.collection.count() == 0:
            self.load_knowledge_base()
    
    def load_knowledge_base(self) -> int:
        """
        Load knowledge base documents into vector store.
        
        Returns:
            Number of documents loaded
        """
        documents = []
        metadatas = []
        ids = []
        
        # Load interview questions
        questions_file = self.knowledge_base_dir / "interview_questions.json"
        if questions_file.exists():
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions_data = json.load(f)
                
                for idx, item in enumerate(questions_data.get("questions", [])):
                    documents.append(f"Question: {item['question']}\nCategory: {item.get('category', 'general')}")
                    metadatas.append({
                        "type": "question",
                        "category": item.get("category", "general"),
                        "difficulty": item.get("difficulty", "medium")
                    })
                    ids.append(f"question_{idx}")
        
        # Load common answers/responses
        responses_file = self.knowledge_base_dir / "common_responses.json"
        if responses_file.exists():
            with open(responses_file, 'r', encoding='utf-8') as f:
                responses_data = json.load(f)
                
                for idx, item in enumerate(responses_data.get("responses", [])):
                    documents.append(f"Scenario: {item['scenario']}\nResponse: {item['response']}")
                    metadatas.append({
                        "type": "response",
                        "scenario_type": item.get("type", "general")
                    })
                    ids.append(f"response_{idx}")
        
        # Load evaluation criteria
        criteria_file = self.knowledge_base_dir / "evaluation_criteria.json"
        if criteria_file.exists():
            with open(criteria_file, 'r', encoding='utf-8') as f:
                criteria_data = json.load(f)
                
                for idx, item in enumerate(criteria_data.get("criteria", [])):
                    documents.append(f"Criterion: {item['name']}\nDescription: {item['description']}")
                    metadatas.append({
                        "type": "criterion",
                        "category": item.get("category", "general")
                    })
                    ids.append(f"criterion_{idx}")
        
        # Add documents to collection if any were loaded
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        return len(documents)
    
    def retrieve_relevant_context(
        self,
        query: str,
        n_results: int = 3,
        filter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: Query text
            n_results: Number of results to return
            filter_type: Optional filter by document type
            
        Returns:
            List of relevant documents with metadata
        """
        where_filter = {"type": filter_type} if filter_type else None
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        
        # Format results
        relevant_docs = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                relevant_docs.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results.get("distances") else 0
                })
        
        return relevant_docs
    
    def get_next_question(
        self,
        conversation_context: str,
        asked_questions: List[str],
        category: Optional[str] = None
    ) -> Optional[str]:
        """
        Get next appropriate question based on context.
        
        Args:
            conversation_context: Current conversation context
            asked_questions: List of already asked questions
            category: Optional category filter
            
        Returns:
            Next question to ask or None
        """
        # Retrieve relevant questions
        filter_dict = {"type": "question"}
        if category:
            filter_dict["category"] = category
        
        results = self.collection.query(
            query_texts=[conversation_context],
            n_results=10,
            where=filter_dict
        )
        
        # Find a question that hasn't been asked
        if results and results["documents"]:
            for doc in results["documents"][0]:
                # Extract question from document
                if "Question:" in doc:
                    question = doc.split("Question:")[1].split("\n")[0].strip()
                    if question not in asked_questions:
                        return question
        
        return None
    
    def get_evaluation_criteria(self, category: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Get evaluation criteria from knowledge base.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of evaluation criteria
        """
        filter_dict = {"type": "criterion"}
        if category:
            filter_dict["category"] = category
        
        results = self.collection.query(
            query_texts=["evaluation criteria"],
            n_results=20,
            where=filter_dict
        )
        
        criteria = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                criteria.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
                })
        
        return criteria
    
    def add_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        doc_id: Optional[str] = None
    ) -> str:
        """
        Add a new document to the knowledge base.
        
        Args:
            content: Document content
            metadata: Document metadata
            doc_id: Optional document ID
            
        Returns:
            Document ID
        """
        if doc_id is None:
            doc_id = f"custom_{self.collection.count()}"
        
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        return doc_id
    
    def clear_knowledge_base(self) -> None:
        """Clear all documents from the knowledge base."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Interview knowledge base"}
        )
