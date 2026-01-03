"""
Detailed tracing system for RLM to capture state at each turn.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List


class RLMDetailedTracer:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.log_file = os.path.join(log_dir, f"rlm_trace_{self.session_id}.jsonl")
        self.turn_count = 0
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    
    def log_turn(self, 
                 iteration: int,
                 messages: List[Dict[str, str]], 
                 response: str,
                 code_blocks: List[str],
                 execution_results: List[str],
                 final_answer: str = None,
                 repl_state: Dict[str, Any] = None,
                 cost_info: Dict[str, float] = None):
        """Log a complete turn of the RLM interaction."""
        turn_data = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "turn": self.turn_count,
            "iteration": iteration,
            "messages": messages,
            "response": response,
            "code_blocks": code_blocks,
            "execution_results": execution_results,
            "final_answer": final_answer,
            "repl_state": repl_state,
            "cost_info": cost_info
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(turn_data) + "\n")
        
        self.turn_count += 1
    
    def log_error(self, error: str, context: str = ""):
        """Log an error with context."""
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "turn": self.turn_count,
            "error": error,
            "context": context
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(error_data) + "\n")
    
    def get_log_path(self):
        """Get the path to the log file."""
        return self.log_file


# Global tracer instance
tracer = RLMDetailedTracer()