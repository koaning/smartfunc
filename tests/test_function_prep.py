import pytest
from pydantic import BaseModel
from smartfunc import _prepare_function_call


class OutputModel(BaseModel):
    result: str


def test_prepare_function_call_basic():
    """Test basic function preparation with simple arguments"""
    def test_func(text: str) -> OutputModel:
        """Process this text: {{ text }}"""
        pass

    docstring, kwargs, return_type = _prepare_function_call(
        test_func, 
        args=("Hello world",), 
        kwargs={}
    )

    assert docstring == "Process this text: Hello world"
    assert kwargs == {"text": "Hello world"}
    assert return_type == OutputModel


def test_prepare_function_call_mixed_args():
    """Test function preparation with both positional and keyword arguments"""
    def test_func(text: str, count: int = 1) -> OutputModel:
        """Process {{ text }} {{ count }} times"""
        pass

    docstring, kwargs, return_type = _prepare_function_call(
        test_func,
        args=("Hello",),
        kwargs={"count": 3}
    )

    assert docstring == "Process Hello 3 times"
    assert kwargs == {"text": "Hello", "count": 3}
    assert return_type == OutputModel


def test_prepare_function_call_default_values():
    """Test function preparation with default values"""
    def test_func(text: str = "default", count: int = 1) -> OutputModel:
        """Process {{ text }} {{ count }} times"""
        pass

    docstring, kwargs, return_type = _prepare_function_call(
        test_func,
        args=(),
        kwargs={}
    )

    assert docstring == "Process default 1 times"
    assert kwargs == {"text": "default", "count": 1}
    assert return_type == OutputModel


def test_prepare_function_call_no_return_type():
    """Test function preparation without return type annotation"""
    def test_func(text: str):
        """Process this: {{ text }}"""
        pass

    docstring, kwargs, return_type = _prepare_function_call(
        test_func,
        args=("Hello",),
        kwargs={}
    )

    assert docstring == "Process this: Hello"
    assert kwargs == {"text": "Hello"}
    assert return_type is None


def test_prepare_function_call_no_docstring():
    """Test function preparation without docstring"""
    def test_func(text: str) -> OutputModel:
        pass

    docstring, kwargs, return_type = _prepare_function_call(
        test_func,
        args=("Hello",),
        kwargs={}
    )

    assert docstring == ""
    assert kwargs == {"text": "Hello"}
    assert return_type == OutputModel


def test_prepare_function_call_invalid_return_type():
    """Test function preparation with invalid return type (non-Pydantic)"""
    def test_func(text: str) -> str:
        """Process this: {{ text }}"""
        pass

    with pytest.raises(AssertionError):
        _prepare_function_call(
            test_func,
            args=("Hello",),
            kwargs={}
        )


def test_prepare_function_call_complex_template():
    """Test function preparation with complex template variables"""
    def test_func(name: str, age: int, hobbies: list) -> OutputModel:
        """{{ name }} is {{ age }} years old and enjoys {{ ', '.join(hobbies) }}."""
        pass

    docstring, kwargs, return_type = _prepare_function_call(
        test_func,
        args=(),
        kwargs={
            "name": "Alice",
            "age": 30,
            "hobbies": ["reading", "coding", "hiking"]
        }
    )

    assert docstring == "Alice is 30 years old and enjoys reading, coding, hiking."
    assert kwargs == {
        "name": "Alice",
        "age": 30,
        "hobbies": ["reading", "coding", "hiking"]
    }
    assert return_type == OutputModel


def test_prepare_function_call_missing_required_arg():
    """Test function preparation with missing required argument"""
    def test_func(text: str, count: int) -> OutputModel:
        """Process {{ text }} {{ count }} times"""
        pass

    with pytest.raises(TypeError):
        _prepare_function_call(
            test_func,
            args=(),
            kwargs={"text": "Hello"}  # missing required 'count' argument
        )
