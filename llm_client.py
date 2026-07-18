"""
NVIDIA Nemotron LLM Client
Reusable module for connecting to NVIDIA's Nemotron LLM API
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, Dict, List

# Load environment variables
load_dotenv()


class NvidiaLLMClient:
    """
    Client for interacting with NVIDIA Nemotron LLM
    
    Usage:
        llm = NvidiaLLMClient()
        response = llm.generate("Your prompt here")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize NVIDIA LLM Client
        
        Args:
            api_key: NVIDIA API key (defaults to NVIDIA_API_KEY env var)
            base_url: API endpoint (defaults to NVIDIA_API_BASE_URL env var)
            model: Model name (defaults to NVIDIA_MODEL env var)
        """
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.base_url = base_url or os.getenv(
            "NVIDIA_API_BASE_URL",
            "https://integrate.api.nvidia.com/v1"
        )
        self.model = model or os.getenv(
            "NVIDIA_MODEL",
            "nvidia/llama-3.3-nemotron-super-49b-v1"
        )
        
        # Validate API key
        if not self.api_key:
            raise ValueError(
                "NVIDIA_API_KEY not found. Please set it in .env file or pass it directly."
            )
        
        # Initialize OpenAI client (works with NVIDIA API too)
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text using NVIDIA Nemotron LLM
        
        Args:
            prompt: User prompt/message
            temperature: Creativity level (0.0-2.0, default 0.6)
            top_p: Diversity (0.0-1.0, default 0.95)
            max_tokens: Maximum tokens in response (default 4096)
            system_prompt: System message for context (optional)
            **kwargs: Additional parameters
        
        Returns:
            Generated text response
            
        Example:
            llm = NvidiaLLMClient()
            response = llm.generate(
                "What is machine learning?",
                temperature=0.7,
                max_tokens=1000
            )
        """
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add user prompt
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                frequency_penalty=kwargs.get("frequency_penalty", 0),
                presence_penalty=kwargs.get("presence_penalty", 0),
                stream=kwargs.get("stream", False)
            )
            
            return completion.choices[0].message.content
        
        except Exception as e:
            raise Exception(f"Error calling NVIDIA API: {str(e)}")
    
    def generate_with_context(
        self,
        prompt: str,
        context: str,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """
        Generate text with additional context
        
        Args:
            prompt: Main question/request
            context: Additional context/information
            temperature: Creativity level
            top_p: Diversity
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
        
        Returns:
            Generated text response
            
        Example:
            context = "My CV: ..."
            prompt = "Generate a professional summary"
            response = llm.generate_with_context(prompt, context)
        """
        full_prompt = f"""CONTEXT:
{context}

REQUEST:
{prompt}"""
        
        return self.generate(
            full_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs
        )
    
    def extract_json(
        self,
        prompt: str,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        **kwargs
    ) -> Dict:
        """
        Generate and parse JSON response
        
        Args:
            prompt: Prompt that should return JSON
            temperature: Creativity level
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
        
        Returns:
            Parsed JSON dictionary
            
        Example:
            prompt = "Return resume data as JSON: {...}"
            data = llm.extract_json(prompt)
        """
        import json
        
        response = self.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Clean up response (remove markdown if present)
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse JSON response: {str(e)}\nResponse: {response}")
    
    def chat_history(
        self,
        messages: List[Dict],
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """
        Continue conversation with chat history
        
        Args:
            messages: List of messages with 'role' and 'content'
            temperature: Creativity level
            top_p: Diversity
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
        
        Returns:
            Generated response
            
        Example:
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ]
            response = llm.chat_history(messages)
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                frequency_penalty=kwargs.get("frequency_penalty", 0),
                presence_penalty=kwargs.get("presence_penalty", 0),
                stream=kwargs.get("stream", False)
            )
            
            return completion.choices[0].message.content
        
        except Exception as e:
            raise Exception(f"Error in chat history: {str(e)}")
    
    def get_model_info(self) -> Dict:
        """Get current model information"""
        return {
            "model": self.model,
            "api_base": self.base_url,
            "api_key_set": bool(self.api_key)
        }


# Convenience functions
def generate(prompt: str, **kwargs) -> str:
    """Quick generation without instantiating class"""
    llm = NvidiaLLMClient()
    return llm.generate(prompt, **kwargs)


def generate_json(prompt: str, **kwargs) -> Dict:
    """Quick JSON generation without instantiating class"""
    llm = NvidiaLLMClient()
    return llm.extract_json(prompt, **kwargs)


def generate_with_context(prompt: str, context: str, **kwargs) -> str:
    """Quick generation with context without instantiating class"""
    llm = NvidiaLLMClient()
    return llm.generate_with_context(prompt, context, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("=== NVIDIA Nemotron LLM Client ===\n")
    
    # Initialize client
    llm = NvidiaLLMClient()
    
    # Example 1: Simple generation
    print("1. Simple Generation:")
    response = llm.generate("What is artificial intelligence in one sentence?")
    print(f"Response: {response}\n")
    
    # Example 2: Generation with context
    print("2. Generation with Context:")
    context = "I am a software engineer with 5 years of experience in Python and AWS."
    prompt = "Write a professional summary for a resume."
    response = llm.generate_with_context(prompt, context)
    print(f"Response: {response}\n")
    
    # Example 3: JSON extraction
    print("3. JSON Extraction:")
    json_prompt = """Return this as JSON:
    Name: John Doe
    Age: 30
    Skills: Python, JavaScript, AWS
    
    Format:
    {
        "name": "...",
        "age": ...,
        "skills": [...]
    }
    """
    data = llm.extract_json(json_prompt)
    print(f"Response: {data}\n")
    
    # Example 4: Model info
    print("4. Model Information:")
    info = llm.get_model_info()
    print(f"Info: {info}")
