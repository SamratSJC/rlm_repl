"""
Utility functions for RLM.
"""

import re
from typing import List, Dict, Optional, Tuple, Any


def find_code_blocks(text: str) -> Optional[List[str]]:
    """Find REPL code blocks in response."""
    pattern = r'```repl\s*\n(.*?)\n```'
    results = []
    
    for match in re.finditer(pattern, text, re.DOTALL):
        code_content = match.group(1).strip()
        results.append(code_content)
    
    return results if results else None


def find_final_answer(text: str) -> Optional[tuple]:
    """Find FINAL() or FINAL_VAR() in response."""
    # Check for FINAL_VAR first
    final_var_pattern = r'^\s*FINAL_VAR\((.*?)\)'
    match = re.search(final_var_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        return ('FINAL_VAR', match.group(1).strip())
    
    # Check for FINAL
    final_pattern = r'^\s*FINAL\((.*?)\)'
    match = re.search(final_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        return ('FINAL', match.group(1).strip())
    
    return None