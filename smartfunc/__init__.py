import datetime 
from functools import wraps
import inspect
from typing import Callable, get_type_hints, Any, Optional, Type
import json
from pydantic import BaseModel
from jinja2 import Template
import llm


def _prepare_function_call(func: Callable, args: tuple, kwargs: dict) -> tuple[str, dict, Optional[Type[BaseModel]]]:
    """Prepares a function call for LLM processing by extracting and validating necessary information.
    
    This internal helper function handles:
    1. Extracting and validating the function's return type (must be Pydantic model if specified)
    2. Processing function arguments and applying defaults
    3. Rendering the function's docstring as a template using the provided arguments
    
    Args:
        func: The function to be processed
        args: Positional arguments passed to the function
        kwargs: Keyword arguments passed to the function
        
    Returns:
        A tuple containing:
        - formatted_docstring: The rendered docstring with arguments applied
        - all_kwargs: Dictionary of all arguments (both positional and keyword)
        - return_type: The expected return type (Pydantic model) or None
        
    Raises:
        AssertionError: If return type is specified but is not a Pydantic BaseModel
    """
    signature = inspect.signature(func)
    docstring = inspect.getdoc(func) or ""
    type_hints = get_type_hints(func)

    # We only support Pydantic now
    if type_hints.get('return', None):
        assert issubclass(type_hints.get('return', None), BaseModel), "Output type must be Pydantic class"

    # Create a dictionary of parameter types
    bound_args = signature.bind(*args, **kwargs)
    bound_args.apply_defaults()  # Apply default values for missing parameters
    all_kwargs = bound_args.arguments

    template = Template(docstring)
    formatted_docstring = template.render(**all_kwargs)

    return formatted_docstring, all_kwargs, type_hints.get('return', None)


def _process_response(response_text: str, return_type: Optional[Type[BaseModel]]) -> Any:
    """Processes the LLM response text based on the expected return type.
    
    This internal helper function handles the conversion of raw LLM response text
    to the appropriate return type.
    
    Args:
        response_text: Raw text response from the LLM
        return_type: Expected return type (Pydantic model) or None
        
    Returns:
        - If return_type is specified: JSON-parsed response converted to the Pydantic model
        - If return_type is None: Raw response text
    """
    if return_type:
        return json.loads(response_text)
    return response_text

def _prepare_debug_info(backend, func: Callable, all_kwargs: dict, formatted_docstring: str, return_type: Optional[Type[BaseModel]] = None) -> dict:
    if return_type:
        return_type = return_type.model_json_schema()
    return {
        "template": func.__doc__,
        "func_name": func.__name__,
        "prompt": formatted_docstring,
        "system": backend.system,
        "template_inputs": all_kwargs,
        "backend_kwargs": backend.kwargs,
        "datetime": datetime.datetime.now().isoformat(),
        "return_type": return_type,
    }

class backend:
    """Synchronous backend decorator for LLM-powered functions.
    
    This class provides a decorator that transforms a function into an LLM-powered
    endpoint. The function's docstring is used as the prompt template, and its
    return type annotation (if specified) determines the expected response format.
    
    Features:
    - Template-based prompt generation from docstring
    - Optional response validation using Pydantic models
    - Synchronous execution
    
    Example:
        @backend("gpt-4")
        def generate_summary(text: str) -> SummaryModel:
            '''Generate a summary of the following text: {{ text }}'''
            pass
    """

    def __init__(self, name, system=None, debug=False, **kwargs):
        """Initialize the backend with specific LLM configuration.
        
        Args:
            name: Name/identifier of the LLM model to use
            system: Optional system prompt for the LLM
            debug: Adds extra information to the output that might help with debugging
            **kwargs: Additional arguments passed to the LLM
        """
        self.model = llm.get_model(name)
        self.system = system
        self.kwargs = kwargs
        self.debug = debug

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            formatted_docstring, all_kwargs, return_type = _prepare_function_call(func, args, kwargs)

            resp = self.model.prompt(
                formatted_docstring,
                system=self.system,
                schema=return_type,
                **self.kwargs
            )
            out = _process_response(resp.text(), return_type)

            if self.debug:
                if isinstance(out, str):
                    out = {"result": out}
                out["_debug"] = _prepare_debug_info(self, func, all_kwargs, formatted_docstring, return_type)

            return out

        return wrapper

    def run(self, func, *args, **kwargs):
        new_func = self(func)
        return new_func(*args, **kwargs)


class async_backend:
    """Asynchronous backend decorator for LLM-powered functions.
    
    Similar to the synchronous `backend` class, but provides asynchronous execution.
    Use this when you need non-blocking LLM operations, typically in async web
    applications or other async contexts.
    
    Features:
    - Async/await support
    - Template-based prompt generation from docstring
    - Optional response validation using Pydantic models
    
    Example:
        @async_backend("gpt-4")
        async def generate_summary(text: str) -> SummaryModel:
            '''Generate a summary of the following text: {{ text }}'''
            pass
    """

    def __init__(self, name, system=None, debug=False, **kwargs):
        """Initialize the async backend with specific LLM configuration.
        
        Args:
            name: Name/identifier of the LLM model to use
            system: Optional system prompt for the LLM
            debug: Adds extra information to the output that might help with debugging
            **kwargs: Additional arguments passed to the LLM
        """
        self.model = llm.get_async_model(name)
        self.system = system
        self.kwargs = kwargs
        self.debug = debug

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            formatted_docstring, all_kwargs, return_type = _prepare_function_call(func, args, kwargs)

            resp = await self.model.prompt(
                formatted_docstring,
                system=self.system,
                schema=return_type,
                **self.kwargs
            )
            text = await resp.text()
            out = _process_response(text, return_type)
        
            if self.debug:
                if isinstance(out, str):
                    out = {"result": out}
                out["_debug"] = _prepare_debug_info(self, func, all_kwargs, formatted_docstring, return_type)

            return out

        return wrapper

    async def run(self, func, *args, **kwargs):
        new_func = self(func)
        return new_func(*args, **kwargs)
    

def get_backend_models():
    for model in llm.get_models():
        print(model.model_id)

def get_async_backend_models():
    for model in llm.get_async_models():
        print(model.model_id)
