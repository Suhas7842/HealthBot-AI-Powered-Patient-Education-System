"""
LLM wrapper with error handling, retries, and structured output support.
Provides robust interface to OpenAI API for HealthBot.
"""

from typing import List, Optional, Type
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from openai import APIError, RateLimitError, APITimeoutError
from healthbot.config import settings
from healthbot.logger import logger


class LLMWrapper:
    """
    Wrapper around ChatOpenAI with error handling and structured outputs.

    Features:
    - Automatic retry with exponential backoff
    - Structured output support via Pydantic
    - Token usage tracking
    - Error handling for common API issues
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize LLM wrapper with support for OpenAI, Groq, or Gemini.

        Args:
            model: Model name (defaults to config setting)
            temperature: Sampling temperature (defaults to config)
            max_tokens: Max tokens to generate (defaults to config)
        """
        self.temperature = temperature if temperature is not None else settings.OPENAI_TEMPERATURE
        self.max_tokens = max_tokens or settings.OPENAI_MAX_TOKENS
        self.provider = settings.LLM_PROVIDER.lower()

        # Initialize based on provider
        if self.provider == "gemini":
            self.model = model or settings.GEMINI_MODEL
            self.llm = ChatGoogleGenerativeAI(
                model=self.model,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,  # Gemini uses max_output_tokens not max_tokens
                google_api_key=settings.GOOGLE_API_KEY
            )
            logger.info(f"LLM initialized: provider=Gemini, model={self.model}, max_output_tokens={self.max_tokens}")
        else:
            # OpenAI or Groq (both use OpenAI-compatible API)
            self.model = model or settings.OPENAI_MODEL
            self.llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                openai_api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL
            )
            logger.info(
                f"LLM initialized: provider={self.provider}, model={self.model}, "
                f"temperature={self.temperature}, max_tokens={self.max_tokens}"
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((APIError, RateLimitError, APITimeoutError)),
        reraise=True
    )
    def invoke(self, messages: List[BaseMessage]) -> str:
        """
        Invoke LLM with retry logic.

        Args:
            messages: List of messages for the conversation

        Returns:
            String response from LLM

        Raises:
            APIError: If API call fails after retries
        """
        try:
            logger.debug(f"Invoking LLM with {len(messages)} messages")

            response = self.llm.invoke(messages)
            content = response.content

            # Log token usage if available
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                logger.info(
                    f"LLM call completed | "
                    f"Input tokens: {usage.get('input_tokens', 0)} | "
                    f"Output tokens: {usage.get('output_tokens', 0)}"
                )

            return content

        except RateLimitError as e:
            logger.warning(f"Rate limit hit, retrying: {e}")
            raise
        except APITimeoutError as e:
            logger.warning(f"API timeout, retrying: {e}")
            raise
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during LLM invocation: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((APIError, RateLimitError, APITimeoutError)),
        reraise=True
    )
    def invoke_structured(
        self,
        messages: List[BaseMessage],
        schema: Type[BaseModel]
    ) -> BaseModel:
        """
        Invoke LLM with structured output using Pydantic schema.

        Args:
            messages: List of messages for the conversation
            schema: Pydantic model class for structured output

        Returns:
            Instance of the Pydantic model with parsed response

        Raises:
            APIError: If API call fails after retries
            ValidationError: If response doesn't match schema
        """
        try:
            logger.debug(f"Invoking LLM with structured output: {schema.__name__}")

            # Use structured output
            structured_llm = self.llm.with_structured_output(schema)
            response = structured_llm.invoke(messages)

            logger.info(f"Structured output generated: {schema.__name__}")
            return response

        except RateLimitError as e:
            logger.warning(f"Rate limit hit, retrying: {e}")
            raise
        except APITimeoutError as e:
            logger.warning(f"API timeout, retrying: {e}")
            raise
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during structured LLM invocation: {e}")
            raise

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text (rough approximation).

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Rough estimate: ~1.3 tokens per word for English
        words = len(text.split())
        return int(words * 1.3)

    def get_config(self) -> dict:
        """
        Get current LLM configuration.

        Returns:
            Dictionary with model configuration
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


def main():
    """Test LLM wrapper functionality."""
    from langchain_core.messages import SystemMessage, HumanMessage
    from healthbot.schemas import MedicalSummary

    wrapper = LLMWrapper()

    print("="*80)
    print("LLM WRAPPER TEST")
    print("="*80)
    print(f"Configuration: {wrapper.get_config()}")

    # Test 1: Basic invocation
    print("\nTest 1: Basic text generation")
    messages = [
        SystemMessage(content="You are a helpful medical education assistant."),
        HumanMessage(content="In one sentence, what is diabetes?")
    ]

    try:
        response = wrapper.invoke(messages)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: Structured output
    print("\nTest 2: Structured output (MedicalSummary)")
    messages = [
        SystemMessage(content="You are a medical education assistant. Return a structured summary."),
        HumanMessage(content="""Create a brief medical summary for diabetes with:
        - title: "Diabetes Mellitus"
        - condition: brief description
        - causes: list of 2 causes
        - symptoms: list of 3 symptoms
        - treatment: list of 2 treatments
        Return as JSON matching the MedicalSummary schema.""")
    ]

    try:
        summary = wrapper.invoke_structured(messages, MedicalSummary)
        print(f"Structured output generated:")
        print(f"  Title: {summary.title}")
        print(f"  Causes: {', '.join(summary.causes)}")
        print(f"  Symptoms: {', '.join(summary.symptoms)}")
    except Exception as e:
        print(f"Error: {e}")

    print("="*80)


if __name__ == "__main__":
    main()
