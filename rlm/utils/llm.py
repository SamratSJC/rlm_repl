"""
LLM client wrapper with cost tracking.
"""

import os
import requests
from typing import Optional, Dict, Tuple


def get_llm_client(api_key: Optional[str], model: str):
    """Factory function to get appropriate LLM client."""
    return LocalClient(api_key, model)


class LocalClient:
    """Local LLM client with cost tracking."""

    # Pricing per 1M tokens (input, output) as of paper
    PRICING = {
        "gpt-5": (2.50, 10.00),
        "gpt-5-mini": (0.15, 0.60),
        "gpt-5-nano": (0.10, 0.40),
        "Nemotron-3-Nano-30B-A3B-IQ4_XS.gguf": (0.05, 0.20),  # Local model pricing
    }

    def __init__(self, api_key: Optional[str], model: str):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # Get API URL from environment variable or default
        self.base_url = os.getenv("RLM_API_URL", "http://localhost:8080/v1")

        # Get available models to verify connection and potentially auto-select model
        try:
            response = requests.get(f"{self.base_url}/models")
            if response.status_code != 200:
                raise ValueError(f"Failed to connect to local LLM server: {response.status_code}")

            # Parse the response to get available models
            models_data = response.json()
            available_models = []

            # Handle different response formats
            if "data" in models_data:  # OpenAI-compatible format
                for model_info in models_data["data"]:
                    available_models.append(model_info.get("id", model_info.get("model", "")))
            elif "models" in models_data:  # Alternative format
                for model_info in models_data["models"]:
                    available_models.append(model_info.get("id", model_info.get("model", "")))
            else:  # Direct array format
                available_models = models_data if isinstance(models_data, list) else []

            # If no specific model was provided, use the first available model
            if (model is None or model == "" or model == "auto") and available_models:
                self.model = available_models[0]
                print(f"Auto-selected model: {self.model}")
            else:
                self.model = model

            # Validate that the selected model exists
            if self.model not in available_models:
                print(f"Warning: Requested model '{self.model}' not found in available models: {available_models}")

        except Exception as e:
            raise ValueError(f"Failed to connect to local LLM server or retrieve models: {e}")

    def completion(self, messages, **kwargs) -> str:
        """Simple completion without cost tracking."""
        try:
            response = self._make_request(messages, **kwargs)
            print(f"Response received: {type(response)}")
            # print(f"Full response: {response}")  # Comment out to avoid spam

            # Handle both chat completion and regular completion formats
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "message" in choice:
                    # Chat completion format
                    content = choice["message"]["content"]
                elif "text" in choice:
                    # Regular completion format
                    content = choice["text"]
                else:
                    raise ValueError(f"Unexpected response format: {choice}")
            else:
                raise ValueError(f"Unexpected response format: {response}")

            return content
        except Exception as e:
            print(f"Error in completion method: {e}")
            import traceback
            traceback.print_exc()
            raise

    def completion_with_cost(
        self,
        messages,
        **kwargs
    ) -> Tuple[str, Dict[str, float]]:
        """Completion with cost tracking."""
        response = self._make_request(messages, **kwargs)

        # Handle both chat completion and regular completion formats
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if "message" in choice:
                # Chat completion format
                content = choice["message"]["content"]
            elif "text" in choice:
                # Regular completion format
                content = choice["text"]
            else:
                raise ValueError(f"Unexpected response format: {choice}")
        else:
            raise ValueError(f"Unexpected response format: {response}")

        # Calculate cost - for local models we'll estimate token counts
        input_tokens = sum(len(msg.get("content", "")) for msg in messages) // 4  # Rough estimate
        output_tokens = len(content) // 4  # Rough estimate
        total_tokens = input_tokens + output_tokens

        # Get pricing
        if self.model in self.PRICING:
            input_price, output_price = self.PRICING[self.model]
        else:
            # Default pricing
            input_price, output_price = 0.05, 0.20  # Lower for local model

        cost = (input_tokens / 1_000_000 * input_price +
                output_tokens / 1_000_000 * output_price)

        cost_info = {
            'cost': cost,
            'tokens': total_tokens,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
        }

        return content, cost_info

    def _make_request(self, messages, **kwargs):
        """Make a request to the local LLM API."""
        headers = {
            "Content-Type": "application/json"
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Check if the local server supports chat completions
        try:
            # Try chat completion endpoint first
            data = {
                "model": self.model,
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7),
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )

            if response.status_code == 200:
                result = response.json()
                # Debug: print the result to see the structure
                # print(f"Chat completion response: {result}")
                return result
            else:
                # If chat completions fail, fall back to completions
                print(f"Chat completions failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Chat completions exception: {e}")
            pass  # Fall through to completions endpoint

        # Fallback to completions endpoint
        # Convert messages to the format expected by local LLM
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"{role.capitalize()}: {content}\n"
        prompt += "Assistant: "

        data = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "stop": kwargs.get("stop", ["User:", "Assistant:"])
        }

        response = requests.post(
            f"{self.base_url}/completions",
            headers=headers,
            json=data
        )

        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code} - {response.text}")

        result = response.json()
        # Debug: print the result to see the structure
        # print(f"Completions response: {result}")
        return result