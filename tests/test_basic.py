import pytest
from unittest.mock import Mock, AsyncMock
from pydantic import BaseModel
from smartfunc import backend, async_backend


class Summary(BaseModel):
    """Test model for structured output."""
    summary: str
    pros: list[str]
    cons: list[str]


def create_mock_client():
    """Create a mock OpenAI client."""
    mock_client = Mock()
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()

    mock_message.content = "test response"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    mock_client.chat.completions.create.return_value = mock_response

    return mock_client, mock_response


def create_async_mock_client():
    """Create a mock async OpenAI client."""
    mock_client = Mock()
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()

    mock_message.content = "test response"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    return mock_client, mock_response


def test_basic_string_output():
    """Test basic function that returns a string."""
    mock_client, mock_response = create_mock_client()

    @backend(mock_client, model="gpt-4o-mini")
    def generate_text(topic: str) -> str:
        """Generate some text."""
        return f"Write about {topic}"

    result = generate_text("testing")

    assert result == "test response"
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args["model"] == "gpt-4o-mini"
    assert call_args["messages"][0]["role"] == "user"
    assert call_args["messages"][0]["content"] == "Write about testing"


def test_structured_output():
    """Test function with structured Pydantic output."""
    mock_client, mock_response = create_mock_client()

    # Mock JSON response
    mock_response.choices[0].message.content = '{"summary": "test", "pros": ["a", "b"], "cons": ["c"]}'

    @backend(mock_client, model="gpt-4o-mini", response_format=Summary)
    def summarize(text: str) -> Summary:
        """Summarize text."""
        return f"Summarize: {text}"

    result = summarize("pokemon")

    assert isinstance(result, Summary)
    assert result.summary == "test"
    assert result.pros == ["a", "b"]
    assert result.cons == ["c"]

    # Verify response_format was set
    call_args = mock_client.chat.completions.create.call_args[1]
    assert "response_format" in call_args
    assert call_args["response_format"]["type"] == "json_schema"


def test_system_prompt():
    """Test that system prompt is correctly passed."""
    mock_client, mock_response = create_mock_client()

    @backend(mock_client, model="gpt-4o-mini", system="You are helpful")
    def generate(prompt: str) -> str:
        return prompt

    result = generate("test")

    call_args = mock_client.chat.completions.create.call_args[1]
    assert len(call_args["messages"]) == 2
    assert call_args["messages"][0]["role"] == "system"
    assert call_args["messages"][0]["content"] == "You are helpful"
    assert call_args["messages"][1]["role"] == "user"


def test_extra_kwargs():
    """Test that extra kwargs are passed to OpenAI API."""
    mock_client, mock_response = create_mock_client()

    @backend(mock_client, model="gpt-4o-mini", temperature=0.7, max_tokens=100)
    def generate(prompt: str) -> str:
        return prompt

    result = generate("test")

    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args["temperature"] == 0.7
    assert call_args["max_tokens"] == 100


def test_function_must_return_string():
    """Test that function must return a string."""
    mock_client, mock_response = create_mock_client()

    @backend(mock_client, model="gpt-4o-mini")
    def bad_function() -> str:
        return 123  # Not a string!

    with pytest.raises(ValueError, match="must return a string prompt"):
        bad_function()


def test_run_method():
    """Test the run method for non-decorator usage."""
    mock_client, mock_response = create_mock_client()

    backend_instance = backend(mock_client, model="gpt-4o-mini")

    def generate(prompt: str) -> str:
        return f"Process: {prompt}"

    result = backend_instance.run(generate, "test")

    assert result == "test response"
    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args["messages"][0]["content"] == "Process: test"


@pytest.mark.asyncio
async def test_async_basic():
    """Test async backend basic functionality."""
    mock_client, mock_response = create_async_mock_client()

    @async_backend(mock_client, model="gpt-4o-mini")
    def generate_text(topic: str) -> str:
        """Generate text async."""
        return f"Write about {topic}"

    result = await generate_text("testing")

    assert result == "test response"
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_async_structured_output():
    """Test async backend with structured output."""
    mock_client, mock_response = create_async_mock_client()

    # Mock JSON response
    mock_response.choices[0].message.content = '{"summary": "async test", "pros": ["fast"], "cons": []}'

    @async_backend(mock_client, model="gpt-4o-mini", response_format=Summary)
    def summarize(text: str) -> Summary:
        """Summarize async."""
        return f"Summarize: {text}"

    result = await summarize("pokemon")

    assert isinstance(result, Summary)
    assert result.summary == "async test"
    assert result.pros == ["fast"]


@pytest.mark.asyncio
async def test_async_run_method():
    """Test the async run method."""
    mock_client, mock_response = create_async_mock_client()

    backend_instance = async_backend(mock_client, model="gpt-4o-mini")

    def generate(prompt: str) -> str:
        return f"Process: {prompt}"

    result = await backend_instance.run(generate, "test")

    assert result == "test response"


def test_multiple_arguments():
    """Test function with multiple arguments."""
    mock_client, mock_response = create_mock_client()

    @backend(mock_client, model="gpt-4o-mini")
    def generate(topic: str, style: str, length: int) -> str:
        return f"Write a {length} word {style} piece about {topic}"

    result = generate("AI", "formal", 500)

    call_args = mock_client.chat.completions.create.call_args[1]
    assert "Write a 500 word formal piece about AI" in call_args["messages"][0]["content"]


def test_complex_prompt_logic():
    """Test that function can have complex prompt generation logic."""
    mock_client, mock_response = create_mock_client()

    @backend(mock_client, model="gpt-4o-mini")
    def smart_generate(items: list[str], include_summary: bool) -> str:
        prompt = "Process these items:\n"
        for i, item in enumerate(items, 1):
            prompt += f"{i}. {item}\n"

        if include_summary:
            prompt += "\nProvide a summary at the end."

        return prompt

    result = smart_generate(["apple", "banana"], True)

    call_args = mock_client.chat.completions.create.call_args[1]
    content = call_args["messages"][0]["content"]
    assert "1. apple" in content
    assert "2. banana" in content
    assert "summary" in content
