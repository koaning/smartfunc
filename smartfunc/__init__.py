from functools import wraps, update_wrapper
from typing import Any, Callable, Optional, Type, Union, List, Dict
import inspect
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
    endpoint. The function can return either:
    - A string that will be used as the user prompt
    - A list of message dictionaries for full conversation control

    The decorator handles calling the LLM and parsing the response.

    Features:
    - Works with any OpenAI SDK-compatible provider (OpenAI, OpenRouter, etc.)
    - Optional structured output validation using Pydantic models
    - Full control over prompt generation using Python
    - Support for multimodal content (images, audio, video via base64)

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
            # Call the function to get the prompt or messages
            result = func(*args, **kwargs)

            # Handle different return types
            if isinstance(result, str):
                # String: build messages with optional system prompt
                messages = []
                if self.system:
                    messages.append({"role": "system", "content": self.system})
                messages.append({"role": "user", "content": result})
            elif isinstance(result, list):
                # List of messages: use directly
                # System prompt is ignored if messages are provided
                messages = result
            else:
                raise ValueError(
                    f"Function {func.__name__} must return either a string prompt "
                    f"or a list of message dictionaries, got {type(result).__name__}"
                )

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

    The function can return either:
    - A string that will be used as the user prompt
    - A list of message dictionaries for full conversation control

    Features:
    - Async/await support for non-blocking operations
    - Works with any OpenAI SDK-compatible provider
    - Optional structured output validation using Pydantic models
    - Support for multimodal content (images, audio, video via base64)

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
            # Call the function to get the prompt or messages
            result = func(*args, **kwargs)

            # Handle different return types
            if isinstance(result, str):
                # String: build messages with optional system prompt
                messages = []
                if self.system:
                    messages.append({"role": "system", "content": self.system})
                messages.append({"role": "user", "content": result})
            elif isinstance(result, list):
                # List of messages: use directly
                # System prompt is ignored if messages are provided
                messages = result
            else:
                raise ValueError(
                    f"Function {func.__name__} must return either a string prompt "
                    f"or a list of message dictionaries, got {type(result).__name__}"
                )

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


class Learnable:
    """Prompt-based learnable function wrapper.

    This class keeps a single canonical prompt string (base_prompt) that can be
    optimized and swapped out during learning. The wrapped function supplies
    template variables for formatting the prompt.
    """

    def __init__(
        self,
        func: Callable,
        client: OpenAI,
        model: str,
        prompt: str,
        output_model: Optional[Union[Type[BaseModel], Type[str]]] = None,
        system: Optional[str] = None,
        **kwargs
    ):
        self.func = func
        self.client = client
        self.model = model
        self.base_prompt = prompt
        self.prompt = prompt
        self.output_model = output_model
        self.system = system
        self.kwargs = kwargs
        update_wrapper(self, func)

    def _render_vars(self, *args, **kwargs) -> Dict[str, Any]:
        result = self.func(*args, **kwargs)
        if result is None:
            sig = inspect.signature(self.func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            variables = dict(bound.arguments)
            variables.pop("self", None)
            return variables
        if isinstance(result, dict):
            return result
        raise ValueError(
            f"Function {self.func.__name__} must return a dict or None, "
            f"got {type(result).__name__}"
        )

    def _call_llm(self, prompt: str):
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})

        call_kwargs = {"model": self.model, "messages": messages, **self.kwargs}

        if self.output_model and self.output_model is not str:
            if not inspect.isclass(self.output_model) or not issubclass(
                self.output_model, BaseModel
            ):
                raise TypeError("output_model must be a BaseModel subclass or str.")
            schema = _disallow_additional_properties(
                self.output_model.model_json_schema()
            )
            call_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": self.output_model.__name__,
                    "schema": schema,
                    "strict": True,
                },
            }

        response = self.client.chat.completions.create(**call_kwargs)
        response_text = response.choices[0].message.content

        if self.output_model and self.output_model is not str:
            return self.output_model.model_validate_json(response_text)
        return response_text

    def __call__(self, *args, **kwargs):
        variables = self._render_vars(*args, **kwargs)
        prompt = self.prompt.format(**variables)
        return self._call_llm(prompt)


class AsyncLearnable:
    """Async prompt-based learnable function wrapper."""

    def __init__(
        self,
        func: Callable,
        client: AsyncOpenAI,
        model: str,
        prompt: str,
        output_model: Optional[Union[Type[BaseModel], Type[str]]] = None,
        system: Optional[str] = None,
        **kwargs
    ):
        self.func = func
        self.client = client
        self.model = model
        self.base_prompt = prompt
        self.prompt = prompt
        self.output_model = output_model
        self.system = system
        self.kwargs = kwargs
        update_wrapper(self, func)

    async def _render_vars(self, *args, **kwargs) -> Dict[str, Any]:
        if inspect.iscoroutinefunction(self.func):
            result = await self.func(*args, **kwargs)
        else:
            result = self.func(*args, **kwargs)
        if result is None:
            sig = inspect.signature(self.func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            variables = dict(bound.arguments)
            variables.pop("self", None)
            return variables
        if isinstance(result, dict):
            return result
        raise ValueError(
            f"Function {self.func.__name__} must return a dict or None, "
            f"got {type(result).__name__}"
        )

    async def _call_llm(self, prompt: str):
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})

        call_kwargs = {"model": self.model, "messages": messages, **self.kwargs}

        if self.output_model and self.output_model is not str:
            if not inspect.isclass(self.output_model) or not issubclass(
                self.output_model, BaseModel
            ):
                raise TypeError("output_model must be a BaseModel subclass or str.")
            schema = _disallow_additional_properties(
                self.output_model.model_json_schema()
            )
            call_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": self.output_model.__name__,
                    "schema": schema,
                    "strict": True,
                },
            }

        response = await self.client.chat.completions.create(**call_kwargs)
        response_text = response.choices[0].message.content

        if self.output_model and self.output_model is not str:
            return self.output_model.model_validate_json(response_text)
        return response_text

    async def __call__(self, *args, **kwargs):
        variables = await self._render_vars(*args, **kwargs)
        prompt = self.prompt.format(**variables)
        return await self._call_llm(prompt)


def learnable(
    client: OpenAI,
    model: str,
    prompt: str,
    output_model: Optional[Union[Type[BaseModel], Type[str]]] = str,
    system: Optional[str] = None,
    **kwargs
):
    """Decorator for creating prompt-based learnable functions."""

    def decorator(func: Callable) -> Learnable:
        return Learnable(
            func=func,
            client=client,
            model=model,
            prompt=prompt,
            output_model=output_model,
            system=system,
            **kwargs,
        )

    return decorator


def async_learnable(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    output_model: Optional[Union[Type[BaseModel], Type[str]]] = str,
    system: Optional[str] = None,
    **kwargs
):
    """Decorator for creating async prompt-based learnable functions."""

    def decorator(func: Callable) -> AsyncLearnable:
        return AsyncLearnable(
            func=func,
            client=client,
            model=model,
            prompt=prompt,
            output_model=output_model,
            system=system,
            **kwargs,
        )

    return decorator


class Pipeline:
    """Sequential pipeline of learnable modules."""

    def __init__(self, *modules: Learnable):
        if not modules:
            raise ValueError("Pipeline requires at least one module.")
        for module in modules:
            if not hasattr(module, "prompt"):
                raise TypeError("All pipeline modules must be learnable.")
        self.modules = list(modules)
        for module in modules:
            name = getattr(module, "__name__", None)
            if name and not hasattr(self, name):
                setattr(self, name, module)

    def __call__(self, *args, **kwargs):
        result = self.modules[0](*args, **kwargs)
        for module in self.modules[1:]:
            result = module(result)
        return result

    def _evaluate(
        self, examples: List[Dict[str, Any]], metric: Callable
    ) -> tuple[float, List[Dict[str, Any]]]:
        scores = []
        predictions = []
        for example in examples:
            if "input" not in example or "output" not in example:
                raise ValueError("Examples must contain 'input' and 'output' keys.")
            prediction = self(example["input"])
            record = {
                "input": example["input"],
                "output": example["output"],
                "prediction": prediction,
            }
            score = metric(record)
            scores.append(score)
            predictions.append(record)
        if not scores:
            return 0.0, predictions
        return sum(scores) / len(scores), predictions

    def _optimizer_prompt(
        self,
        module: Learnable,
        predictions: List[Dict[str, Any]],
        module_index: int,
    ) -> str:
        lines = [
            "You are improving a prompt for a step in a multi-step pipeline.",
            f"Step: {module_index + 1} of {len(self.modules)}",
            "",
            "Current prompt:",
            module.prompt,
            "",
            "Examples (input -> expected output, current prediction):",
        ]
        for item in predictions:
            lines.append(f"- input: {item['input']}")
            lines.append(f"  expected_output: {item['output']}")
            lines.append(f"  prediction: {item['prediction']}")
        lines.extend(
            [
                "",
                "Provide an improved prompt. Return only the prompt text.",
            ]
        )
        return "\n".join(lines)

    def _propose_prompt(
        self,
        module: Learnable,
        predictions: List[Dict[str, Any]],
        module_index: int,
        n_candidates: int,
    ) -> List[str]:
        request = self._optimizer_prompt(module, predictions, module_index)
        messages = [{"role": "user", "content": request}]

        candidates = []
        for _ in range(n_candidates):
            response = module.client.chat.completions.create(
                model=module.model, messages=messages
            )
            candidates.append(response.choices[0].message.content.strip())
        return candidates

    def learn(
        self,
        examples: List[Dict[str, Any]],
        metric: Callable[[Dict[str, Any]], float],
        steps: int = 3,
        candidates: int = 1,
    ):
        if metric is None:
            raise ValueError("metric must be provided for learning.")
        if candidates < 1:
            raise ValueError("candidates must be >= 1.")
        for _ in range(steps):
            for idx, module in enumerate(self.modules):
                current_prompt = module.prompt
                best_score, baseline_predictions = self._evaluate(examples, metric)
                best_prompt = current_prompt

                for proposal in self._propose_prompt(
                    module, baseline_predictions, idx, candidates
                ):
                    module.prompt = proposal
                    proposal_score, _ = self._evaluate(examples, metric)
                    if proposal_score > best_score:
                        best_score = proposal_score
                        best_prompt = proposal

                module.prompt = best_prompt
        return self

