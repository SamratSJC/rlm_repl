"""
Recursive Language Model with REPL environment.
Implements the core RLM algorithm from the paper.
"""

from typing import Dict, List, Optional, Any, Union
import re

from rlm import RLM
from rlm.repl import REPLEnv
from rlm.utils.tracing import tracer


class RLM_REPL(RLM):
    """
    RLM implementation using REPL environment.
    Context is stored externally in REPL, not passed to model directly.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5",
        recursive_model: str = "gpt-5-mini",
        max_iterations: int = 20,
        depth: int = 0,
    ):
        """
        Initialize RLM with REPL.
        
        Args:
            api_key: API key for LLM provider
            model: Root LLM model name
            recursive_model: Sub-LLM model name for recursive calls
            max_iterations: Maximum number of root LLM iterations
            depth: Current recursion depth (0 for root)
        """
        self.api_key = api_key
        self.model = model
        self.recursive_model = recursive_model
        self.max_iterations = max_iterations
        self.depth = depth
        
        # Initialize LLM client (will be imported from utils)
        from rlm.utils.llm import get_llm_client
        self.llm = get_llm_client(api_key, model)
        
        # Initialize cost tracking
        self._root_llm_cost = 0.0
        self._sub_llm_cost = 0.0
        self._root_llm_tokens = 0
        self._sub_llm_tokens = 0
        self._root_llm_calls = 0
        self._sub_llm_calls = 0
        
        # State
        self.repl_env: Optional[REPLEnv] = None
        self.messages: List[Dict[str, str]] = []
        self.query: Optional[str] = None
    
    def _setup_context(
        self,
        context: Union[List[str], str, List[Dict[str, str]]],
        query: str
    ):
        """Setup the REPL environment with context."""
        print("_setup_context called")
        self.query = query
        self.messages = []

        # Build system prompt
        print("Building system prompt...")
        from rlm.utils.prompts import build_system_prompt
        self.messages = build_system_prompt(self.model)

        # Convert context for REPL
        print("Converting context...")
        context_data, context_str = self._convert_context(context)
        print(f"Context converted: context_data type={type(context_data)}, context_str type={type(context_str)}")

        # Get context metadata
        print("Getting context metadata...")
        context_type, context_lengths, context_total_length = self._get_context_metadata(
            context, context_data, context_str
        )
        print(f"Metadata: type={context_type}, lengths={context_lengths[:3]}..., total={context_total_length}")

        # Create llm_query function for recursive calls
        def llm_query_fn(prompt: str) -> str:
            return self._recursive_llm_call(prompt)

        # Initialize REPL with context
        print("Initializing REPL environment...")
        self.repl_env = REPLEnv(
            llm_query_fn=llm_query_fn,
            context_json=context_data,
            context_str=context_str,
        )
        print("REPL environment initialized")

        # Add context metadata to initial message
        print("Adding context metadata to messages...")
        from rlm.utils.prompts import add_context_metadata
        self.messages = add_context_metadata(
            self.messages,
            context_type,
            context_lengths,
            context_total_length
        )
        print("Context metadata added")
    
    def _convert_context(self, context):
        """Convert context to appropriate format for REPL."""
        if isinstance(context, dict):
            return context, None
        elif isinstance(context, str):
            return None, context
        elif isinstance(context, list):
            if len(context) > 0 and isinstance(context[0], dict):
                if "content" in context[0]:
                    return [msg.get("content", "") for msg in context], None
                else:
                    return context, None
            else:
                return context, None
        else:
            return context, None
    
    def _get_context_metadata(self, context, context_data, context_str):
        """Get metadata about context for prompting."""
        if context_str is not None:
            context_type = "str"
            context_total_length = len(context_str)
            context_lengths = [context_total_length]
        elif context_data is not None:
            if isinstance(context_data, list):
                context_type = "list"
                context_lengths = [len(str(item)) for item in context_data]
                context_total_length = sum(context_lengths)
            elif isinstance(context_data, dict):
                context_type = "dict"
                context_lengths = [len(str(context_data))]
                context_total_length = context_lengths[0]
            else:
                context_type = type(context_data).__name__
                context_lengths = [len(str(context_data))]
                context_total_length = context_lengths[0]
        else:
            context_type = "unknown"
            context_lengths = [0]
            context_total_length = 0
        
        return context_type, context_lengths, context_total_length
    
    def _recursive_llm_call(self, prompt: str) -> str:
        """
        Make a recursive LLM call (sub-LLM).
        In paper, depth=1, so sub-LLMs are regular LLMs.
        """
        from rlm.utils.llm import get_llm_client
        
        # Use recursive model for sub-calls
        sub_llm = get_llm_client(self.api_key, self.recursive_model)
        
        # Format prompt as message
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        
        # Make call and track cost
        response, cost_info = sub_llm.completion_with_cost(messages)
        
        self._sub_llm_cost += cost_info['cost']
        self._sub_llm_tokens += cost_info['tokens']
        self._sub_llm_calls += 1
        
        return response
    
    def _find_code_blocks(self, text: str) -> Optional[List[str]]:
        """Find REPL code blocks in response."""
        pattern = r'```repl\s*\n(.*?)\n```'
        results = []
        
        for match in re.finditer(pattern, text, re.DOTALL):
            code_content = match.group(1).strip()
            results.append(code_content)
        
        return results if results else None
    
    def _find_final_answer(self, text: str) -> Optional[tuple]:
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
    
    def _execute_code(self, code: str) -> str:
        """Execute code in REPL and return formatted result."""
        result = self.repl_env.code_execution(code)
        
        # Format result for display
        result_parts = []
        
        if result.stdout:
            result_parts.append(f"\n{result.stdout}")
        
        if result.stderr:
            result_parts.append(f"\nError: {result.stderr}")
        
        # Show some key variables
        important_vars = {}
        for key, value in result.locals.items():
            if not key.startswith('_') and key not in ['__builtins__', '__name__', '__doc__']:
                try:
                    if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                        if isinstance(value, str) and len(value) > 100:
                            important_vars[key] = f"'{value[:100]}...'"
                        else:
                            important_vars[key] = repr(value)
                except:
                    important_vars[key] = f"<{type(value).__name__}>"
        
        if important_vars:
            result_parts.append(f"\nREPL variables: {list(important_vars.keys())}")
        
        formatted = "\n".join(result_parts) if result_parts else "No output"
        
        # Truncate if too long
        max_length = 100000
        if len(formatted) > max_length:
            formatted = formatted[:max_length] + "..."
        
        return formatted
    
    def _process_code_execution(self, response: str) -> List[Dict[str, str]]:
        """Process code blocks in response and update messages."""
        messages, _ = self._process_code_execution_with_results(response)
        return messages

    def _process_code_execution_with_results(self, response: str):
        """Process code blocks in response and update messages, returning both messages and execution results."""
        code_blocks = self._find_code_blocks(response)
        execution_results = []

        if code_blocks:
            for code in code_blocks:
                execution_result = self._execute_code(code)
                execution_results.append(execution_result)

                # Add to messages
                self.messages.append({
                    "role": "user",
                    "content": f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{execution_result}"
                })

        return self.messages, execution_results
    
    def _check_final_answer(self, response: str) -> Optional[str]:
        """Check if response contains final answer."""
        result = self._find_final_answer(response)
        if result is None:
            return None
        
        answer_type, content = result
        
        if answer_type == 'FINAL':
            return content
        elif answer_type == 'FINAL_VAR':
            # Get variable from REPL
            variable_name = content.strip().strip('"').strip("'").strip('\n').strip('\r')
            
            if variable_name in self.repl_env.locals:
                return str(self.repl_env.locals[variable_name])
            else:
                return None
        
        return None
    
    def completion(
        self, 
        context: Union[List[str], str, List[Dict[str, str]]], 
        query: str
    ) -> str:
        """
        Generate completion using RLM with REPL.

        Args:
            context: Context to process (can be arbitrarily long)
            query: Query to answer

        Returns:
            Final answer string
        """
        print("Starting RLM completion...")
        # Setup REPL with context
        print("Setting up context...")
        self._setup_context(context, query)
        print("Context setup complete")
        
        # Main iteration loop
        for iteration in range(self.max_iterations):
            # Build prompt for this iteration
            from rlm.utils.prompts import next_action_prompt
            user_prompt = next_action_prompt(query, iteration)
            
            # Get model response
            print("Getting model response...")
            response, cost_info = self.llm.completion_with_cost(
                self.messages + [user_prompt]
            )
            print(f"Response received: {response[:100]}...")  # First 100 chars

            # Track root LLM cost
            self._root_llm_cost += cost_info['cost']
            self._root_llm_tokens += cost_info['tokens']
            self._root_llm_calls += 1

            # Process code execution if present
            print("Finding code blocks...")
            code_blocks = self._find_code_blocks(response)
            print(f"Found {len(code_blocks) if code_blocks else 0} code blocks")

            execution_results = []
            if code_blocks:
                print("Processing code execution...")
                # We need to modify _process_code_execution to return execution results
                self.messages, execution_results = self._process_code_execution_with_results(response)
            else:
                print("No code blocks found, adding as assistant message")
                # No code blocks, add as assistant message
                self.messages.append({
                    "role": "assistant",
                    "content": "You responded with:\n" + response
                })

            # Log the turn with detailed information
            repl_state = {
                'context_loaded': 'context' in (self.repl_env.locals if self.repl_env else {}),
                'local_vars': list(self.repl_env.locals.keys()) if self.repl_env else [],
                'globals': list(self.repl_env.globals.keys()) if self.repl_env else []
            } if self.repl_env else {}

            tracer.log_turn(
                iteration=iteration,
                messages=self.messages,
                response=response,
                code_blocks=code_blocks or [],
                execution_results=execution_results,
                repl_state=repl_state,
                cost_info=cost_info
            )
            
            # Check for final answer in the response
            final_answer = self._check_final_answer(response)
            if final_answer:
                # Log the final answer
                tracer.log_turn(
                    iteration=iteration,
                    messages=self.messages,
                    response=response,
                    code_blocks=code_blocks or [],
                    execution_results=execution_results,
                    final_answer=final_answer,
                    repl_state={
                        'context_loaded': 'context' in (self.repl_env.locals if self.repl_env else {}),
                        'local_vars': list(self.repl_env.locals.keys()) if self.repl_env else [],
                        'globals': list(self.repl_env.globals.keys()) if self.repl_env else []
                    } if self.repl_env else {},
                    cost_info=cost_info
                )
                return final_answer

            # Also check if any variable in the REPL environment contains a final answer
            # This handles cases where code execution created a variable with the answer
            if self.repl_env and hasattr(self.repl_env, 'locals'):
                for var_name, var_value in self.repl_env.locals.items():
                    if isinstance(var_value, str):
                        # Check if this variable value is a final answer
                        if var_value.startswith('FINAL(') and var_value.endswith(')'):
                            actual_answer = var_value[6:-1]  # Extract content between FINAL(...)
                            tracer.log_turn(
                                iteration=iteration,
                                messages=self.messages,
                                response=response,
                                code_blocks=code_blocks or [],
                                execution_results=execution_results,
                                final_answer=actual_answer,
                                repl_state={
                                    'context_loaded': 'context' in (self.repl_env.locals if self.repl_env else {}),
                                    'local_vars': list(self.repl_env.locals.keys()) if self.repl_env else [],
                                    'globals': list(self.repl_env.globals.keys()) if self.repl_env else []
                                } if self.repl_env else {},
                                cost_info=cost_info
                            )
                            return actual_answer
        
        # If no final answer after max iterations, force one
        from rlm.utils.prompts import next_action_prompt
        final_prompt = next_action_prompt(query, self.max_iterations, final_answer=True)
        self.messages.append(final_prompt)

        response, cost_info = self.llm.completion_with_cost(self.messages)
        self._root_llm_cost += cost_info['cost']
        self._root_llm_tokens += cost_info['tokens']
        self._root_llm_calls += 1

        # Log the final response
        tracer.log_turn(
            iteration=self.max_iterations,
            messages=self.messages,
            response=response,
            code_blocks=[],
            execution_results=[],
            final_answer=response,
            repl_state={
                'context_loaded': 'context' in (self.repl_env.locals if self.repl_env else {}),
                'local_vars': list(self.repl_env.locals.keys()) if self.repl_env else [],
                'globals': list(self.repl_env.globals.keys()) if self.repl_env else []
            } if self.repl_env else {},
            cost_info=cost_info
        )

        return response
    
    def cost_summary(self) -> Dict[str, Any]:
        """Get cost summary for this completion."""
        return {
            'total_cost': self._root_llm_cost + self._sub_llm_cost,
            'root_llm_cost': self._root_llm_cost,
            'sub_llm_cost': self._sub_llm_cost,
            'root_llm_tokens': self._root_llm_tokens,
            'sub_llm_tokens': self._sub_llm_tokens,
            'root_llm_calls': self._root_llm_calls,
            'sub_llm_calls': self._sub_llm_calls,
        }
    
    def reset(self):
        """Reset RLM state."""
        self.repl_env = None
        self.messages = []
        self.query = None
        self._root_llm_cost = 0.0
        self._sub_llm_cost = 0.0
        self._root_llm_tokens = 0
        self._sub_llm_tokens = 0
        self._root_llm_calls = 0
        self._sub_llm_calls = 0