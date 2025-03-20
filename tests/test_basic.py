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


@pytest.mark.parametrize("text", ["Hello, world!", "Hello, programmer!"])
def test_basic_output(text):
    """Ensure that we can also put the returned string in the output"""
    @backend("markov")
    def generate_summary(t):
        """Generate a summary of the following text: {{ t }}"""
        return "dinosaurhead"

    assert "dinosaurhead" in generate_summary(text) 
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
    assert result["_debug"]["template_inputs"] == {"t": "Hello, world!"}
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
    assert result["_debug"]["template_inputs"] == {"a": "Hello", "b": "world", "c": "!"}
    assert result["_debug"]["system"] == "You are a helpful assistant."
    assert result["result"]


def test_debug_info_keys():
    """Test that all expected debug info keys are present with correct types"""
    @backend("markov", debug=True, system="You are a helpful assistant.")
    def generate_summary(text):
        """Generate a summary of the following text: {{ text }}"""
        return "dinosaurhead"

    result = generate_summary("Hello, world!")
    debug_info = result["_debug"]
    
    # Check all expected keys are present
    expected_keys = {
        "template": str,  # Original docstring template
        "func_name": str,  # Name of the function
        "prompt": str,    # Rendered prompt
        "system": str,    # System prompt
        "template_inputs": dict,  # Arguments used in template
        "backend_kwargs": dict,   # Backend configuration
        "datetime": str,   # ISO format datetime
        "return_type": type(None)  # None since no return type specified
    }
    
    for key, expected_type in expected_keys.items():
        assert key in debug_info, f"Missing debug key: {key}"
        assert isinstance(debug_info[key], expected_type), f"Incorrect type for {key}: expected {expected_type}, got {type(debug_info[key])}"
    
    # Check specific values
    assert debug_info["template"] == "Generate a summary of the following text: {{ text }}"
    assert debug_info["func_name"] == "generate_summary"
    assert debug_info["prompt"] == "Generate a summary of the following text: Hello, world! dinosaurhead"
    assert debug_info["system"] == "You are a helpful assistant."
    assert debug_info["template_inputs"] == {"text": "Hello, world!"}
    assert debug_info["return_type"] is None 