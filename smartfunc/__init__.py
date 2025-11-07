from functools import wraps
from typing import Any, Callable, Optional, Type, Union
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI


def _disallow_additional_properties(schema: Any) -> Any:
    """Ensure every object schema explicitly forbids unknown properties (OpenAI requirement)."""
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
            props = schema.get("properties")
            if isinstance(props, dict):
                for value in props.values():
                    _disallow_additional_properties(value)

        items = schema.get("items")
        if isinstance(items, dict) or isinstance(items, list):
            _disallow_additional_properties(items)

        for keyword in ("allOf", "anyOf", "oneOf"):
            subschema = schema.get(keyword)
            if isinstance(subschema, list):
                for item in subschema:
                    _disallow_additional_properties(item)
            elif isinstance(subschema, dict):
                _disallow_additional_properties(subschema)

        not_schema = schema.get("not")
        if isinstance(not_schema, dict):
            _disallow_additional_properties(not_schema)

        for defs_key in ("definitions", "$defs"):
            defs = schema.get(defs_key)
            if isinstance(defs, dict):
                for value in defs.values():
                    _disallow_additional_properties(value)

    elif isinstance(schema, list):
        for item in schema:
            _disallow_additional_properties(item)

    return schema


class backend:
    """Synchronous backend decorator for LLM-powered functions.

    This class provides a decorator that transforms a function into an LLM-powered
    endpoint. The function should return a string that will be used as the prompt,
    and the decorator handles calling the LLM and parsing the response.

    Features:
    - Works with any OpenAI SDK-compatible provider (OpenAI, OpenRouter, etc.)
    - Optional structured output validation using Pydantic models
    - Full control over prompt generation using Python

    Example:
        from openai import OpenAI
        from pydantic import BaseModel

        client = OpenAI()

        class Summary(BaseModel):
            summary: str
            pros: list[str]

        @backend(client, model="gpt-4o-mini", response_format=Summary)
        def generate_summary(text: str) -> Summary:
            '''Generate a summary of the following text.'''
            return f"Summarize this text: {text}"

        result = generate_summary("Some text here")
        print(result.summary)
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        response_format: Optional[Type[BaseModel]] = None,
        system: Optional[str] = None,
        **kwargs
    ):
        """Initialize the backend with specific LLM configuration.

        Args:
            client: OpenAI client instance (or compatible client)
            model: Name/identifier of the model to use (e.g., "gpt-4o-mini")
            response_format: Optional Pydantic model for structured output
            system: Optional system prompt for the LLM
            **kwargs: Additional arguments passed to the OpenAI API (e.g., temperature, max_tokens)
        """
        self.client = client
        self.model = model
        self.response_format = response_format
        self.system = system
        self.kwargs = kwargs

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Call the function to get the prompt
            prompt = func(*args, **kwargs)

            if not isinstance(prompt, str):
                raise ValueError(
                    f"Function {func.__name__} must return a string prompt, "
                    f"got {type(prompt).__name__}"
                )

            # Build messages array
            messages = []
            if self.system:
                messages.append({"role": "system", "content": self.system})
            messages.append({"role": "user", "content": prompt})

            # Prepare API call kwargs
            call_kwargs = {
                "model": self.model,
                "messages": messages,
                **self.kwargs
            }

            # Add structured output if specified
            if self.response_format:
                schema = _disallow_additional_properties(
                    self.response_format.model_json_schema()
                )
                call_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.response_format.__name__,
                        "schema": schema,
                        "strict": True
                    }
                }

            # Call OpenAI API
            response = self.client.chat.completions.create(**call_kwargs)
            response_text = response.choices[0].message.content

            # Parse response
            if self.response_format:
                return self.response_format.model_validate_json(response_text)
            else:
                return response_text

        return wrapper

    def run(self, func: Callable, *args, **kwargs):
        """Run a function through the backend without using it as a decorator.

        Args:
            func: The function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            The result from the LLM (parsed according to response_format)
        """
        decorated_func = self(func)
        return decorated_func(*args, **kwargs)


class async_backend:
    """Asynchronous backend decorator for LLM-powered functions.

    Similar to the synchronous `backend` class, but provides asynchronous execution.
    Use this when you need non-blocking LLM operations, typically in async web
    applications or for concurrent processing.

    Features:
    - Async/await support for non-blocking operations
    - Works with any OpenAI SDK-compatible provider
    - Optional structured output validation using Pydantic models

    Example:
        from openai import AsyncOpenAI
        from pydantic import BaseModel
        import asyncio

        client = AsyncOpenAI()

        class Summary(BaseModel):
            summary: str

        @async_backend(client, model="gpt-4o-mini", response_format=Summary)
        async def generate_summary(text: str) -> Summary:
            '''Generate a summary.'''
            return f"Summarize: {text}"

        result = asyncio.run(generate_summary("text"))
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        response_format: Optional[Type[BaseModel]] = None,
        system: Optional[str] = None,
        **kwargs
    ):
        """Initialize the async backend with specific LLM configuration.

        Args:
            client: AsyncOpenAI client instance (or compatible async client)
            model: Name/identifier of the model to use
            response_format: Optional Pydantic model for structured output
            system: Optional system prompt for the LLM
            **kwargs: Additional arguments passed to the OpenAI API
        """
        self.client = client
        self.model = model
        self.response_format = response_format
        self.system = system
        self.kwargs = kwargs

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Call the function to get the prompt
            prompt = func(*args, **kwargs)

            if not isinstance(prompt, str):
                raise ValueError(
                    f"Function {func.__name__} must return a string prompt, "
                    f"got {type(prompt).__name__}"
                )

            # Build messages array
            messages = []
            if self.system:
                messages.append({"role": "system", "content": self.system})
            messages.append({"role": "user", "content": prompt})

            # Prepare API call kwargs
            call_kwargs = {
                "model": self.model,
                "messages": messages,
                **self.kwargs
            }

            # Add structured output if specified
            if self.response_format:
                schema = _disallow_additional_properties(
                    self.response_format.model_json_schema()
                )
                call_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.response_format.__name__,
                        "schema": schema,
                        "strict": True
                    }
                }

            # Call OpenAI API
            response = await self.client.chat.completions.create(**call_kwargs)
            response_text = response.choices[0].message.content

            # Parse response
            if self.response_format:
                return self.response_format.model_validate_json(response_text)
            else:
                return response_text

        return wrapper

    async def run(self, func: Callable, *args, **kwargs):
        """Run a function through the backend without using it as a decorator.

        Args:
            func: The function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            The result from the LLM (parsed according to response_format)
        """
        decorated_func = self(func)
        return await decorated_func(*args, **kwargs)
