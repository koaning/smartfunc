from pydantic import BaseModel
import pytest 

from smartfunc import backend, async_backend


@pytest.mark.parametrize("text", ["Hello, world!", "Hello, programmer!"])
def test_basic(text):
    """Test basic function call with the markov backend"""
    @backend("markov")
    def generate_summary(t):
        """Generate a summary of the following text: {{ t }}"""
        pass

    assert text in generate_summary(text) 


def test_schema_error():
    """The markov backend does not support schemas, error should be raised"""
    with pytest.raises(ValueError):
        class OutputModel(BaseModel):
            result: str

        @backend("markov", delay=0, length=10)
        def generate_summary(t) -> OutputModel:
            """Generate a summary of the following text: {{ t }}"""
            pass

        generate_summary("Hello, world!")


def test_debug_mode_1():
    """Test that debug mode works when we do not pass a type"""
    @backend("markov", debug=True, system="You are a helpful assistant.")
    def generate_summary(t):
        """Generate a summary of the following text: {{ t }}"""
        pass

    result = generate_summary("Hello, world!")
    
    assert isinstance(result, dict)
    assert result["_debug"]["prompt"] == "Generate a summary of the following text: Hello, world!"
    assert result["_debug"]["kwargs"] == {"t": "Hello, world!"}
    assert result["_debug"]["system"] == "You are a helpful assistant."
    assert result["result"]


def test_debug_mode_2():
    """Test that debug mode works with multiple arguments"""
    @backend("markov", debug=True, system="You are a helpful assistant.")
    def generate_summary(a, b, c):
        """Generate a summary of the following text: {{ a }} {{ b }} {{ c }}"""
        pass

    result = generate_summary("Hello", "world", "!")
    
    assert isinstance(result, dict)
    assert result["_debug"]["prompt"] == "Generate a summary of the following text: Hello world !"
    assert result["_debug"]["kwargs"] == {"a": "Hello", "b": "world", "c": "!"}
    assert result["_debug"]["system"] == "You are a helpful assistant."
    assert result["result"]
    