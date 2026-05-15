import os
from typing import Any
from langchain_openai import ChatOpenAI

def get_llm(temperature: float = 0.2) -> Any:
    """Returns a chat model based on environment variables."""
    
    # 1. Try Gemini
    if os.environ.get("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.environ.get("LLM_MODEL", "gemini-3.1-flash-lite"),
            temperature=temperature,
        )

    # 2. Try Local (Ollama via OpenAI-compatible API)
    base_url = os.environ.get("LLM_BASE_URL", "")
    if "localhost" in base_url or "127.0.0.1" in base_url:
        return ChatOpenAI(
            model=os.environ.get("LLM_MODEL", "llama3.1"),
            base_url=base_url,
            api_key="ollama",  # Placeholder
            temperature=temperature,
        )

    # 3. Default: OpenRouter / OpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No LLM API key found. Please set GOOGLE_API_KEY for Gemini, "
            "or OPENROUTER_API_KEY for OpenRouter, or LLM_BASE_URL for local models."
        )
        
    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
        base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
        temperature=temperature,
    )
