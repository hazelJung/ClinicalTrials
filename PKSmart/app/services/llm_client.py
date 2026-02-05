from abc import ABC, abstractmethod
import os


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        pass


class OpenAIClient(LLMClient):
    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=8192,
            temperature=0.3,
        )
        return response.choices[0].message.content


class GeminiClient(LLMClient):
    """Google Gemini API Client"""
    def __init__(self):
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.model = genai.GenerativeModel(self.model_name)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Combine system and user prompts for Gemini
        full_prompt = f"""[SYSTEM INSTRUCTIONS]
{system_prompt}

[USER REQUEST]
{user_prompt}"""
        
        response = self.model.generate_content(
            full_prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 65536,  # Increased for long IND documents
            }
        )
        return response.text


class OllamaClient(LLMClient):
    def __init__(self):
        self.model = os.getenv("OLLAMA_MODEL", "gemma3:12b")
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            import ollama
        except ImportError:
            raise ImportError("ollama package is not installed. Please run: pip install ollama")

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response["message"]["content"]


def get_llm_client() -> LLMClient:
    """
    Get LLM client based on LLM_PROVIDER environment variable.
    
    Supported providers:
    - 'ollama': Local Ollama (default)
    - 'openai': OpenAI API
    - 'gemini': Google Gemini API (recommended for complex documents)
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    if provider == "openai":
        return OpenAIClient()
    elif provider == "gemini":
        return GeminiClient()
    else:
        return OllamaClient()
