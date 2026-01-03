"""
Abstract base class for Recursive Language Models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union

class RLM(ABC):
    """Abstract base class defining the RLM interface."""
    
    @abstractmethod
    def completion(
        self, 
        context: Union[List[str], str, List[Dict[str, str]]], 
        query: str
    ) -> str:
        """
        Generate a completion for the given query and context.
        
        Args:
            context: The context to process (can be very long)
            query: The query/question to answer
            
        Returns:
            The final answer as a string
        """
        pass
    
    @abstractmethod
    def cost_summary(self) -> Dict[str, Any]:
        """
        Get a summary of costs incurred during completion.
        
        Returns:
            Dictionary with cost breakdown:
                - total_cost: Total cost in USD
                - root_llm_cost: Cost of root LLM calls
                - sub_llm_cost: Cost of sub-LLM calls
                - root_llm_tokens: Token count for root
                - sub_llm_tokens: Token count for sub-LLMs
                - root_llm_calls: Number of root LLM calls
                - sub_llm_calls: Number of sub-LLM calls
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the RLM state for a new task."""
        pass