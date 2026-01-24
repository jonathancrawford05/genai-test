"""
Base agent class for multi-agent RAG system.

Provides common functionality:
- LLM interaction via Ollama
- Conversation history management
- Prompt formatting
- Response parsing
"""
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import ollama


@dataclass
class Message:
    """Single message in conversation history."""
    role: str  # "system", "user", or "assistant"
    content: str


class BaseAgent:
    """
    Abstract base class for RAG agents.

    Handles:
    - LLM communication
    - Conversation history
    - Common utilities
    """

    def __init__(self, model: str = "llama3.2", temperature: float = 0.0):
        """
        Initialize agent.

        Args:
            model: Ollama model name
            temperature: LLM temperature (0.0 = deterministic)
        """
        self.model = model
        self.temperature = temperature
        self.conversation_history: List[Message] = []

    def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        reset_history: bool = False
    ) -> str:
        """
        Call LLM with prompt and optional system message.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response
            reset_history: Clear conversation history before call

        Returns:
            LLM response text
        """
        if reset_history:
            self.conversation_history = []

        # Build messages
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation history
        for msg in self.conversation_history:
            messages.append({"role": msg.role, "content": msg.content})

        # Add current user prompt
        messages.append({"role": "user", "content": prompt})

        # Call Ollama
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": max_tokens
                }
            )

            response_text = response["message"]["content"].strip()

            # Update conversation history
            self.conversation_history.append(Message(role="user", content=prompt))
            self.conversation_history.append(Message(role="assistant", content=response_text))

            return response_text

        except Exception as e:
            raise RuntimeError(
                f"LLM call failed: {e}\n"
                f"Make sure Ollama is running and {self.model} is installed."
            )

    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []

    def get_conversation_history(self) -> List[Message]:
        """Get conversation history."""
        return self.conversation_history.copy()
