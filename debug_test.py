"""
Simple test to debug the RLM issue
"""

from rlm.rlm_repl import RLM_REPL

def test_simple():
    # Create a small context to test
    context = "This is a test context with some text. " * 100
    context += "The magic number is 123456789."
    context += " More text to make it longer. " * 100
    
    print(f"Context length: {len(context)}")
    
    rlm = RLM_REPL(
        model="Qwen3-Coder-REAP-25B-A3B.Q5_K_M.gguf",
        recursive_model="Qwen3-Coder-REAP-25B-A3B.Q5_K_M.gguf",
        max_iterations=5
    )
    
    try:
        result = rlm.completion(
            context=context,
            query="What is the magic number in the context?"
        )
        if result is None:
            print("Result: RLM reached max iterations without finding a final answer.")
        else:
            print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple()