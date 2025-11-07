import pytest
from pydantic import BaseModel
from smartfunc import backend, async_backend


class Summary(BaseModel):
    """Test model for structured output."""
    summary: str
    pros: list[str]
    cons: list[str]


def test_basic_string_output(mock_client_factory):
    """Test basic function that returns a string."""
    client = mock_client_factory()

    @backend(client, model="gpt-4o-mini")
    def generate_text(topic: str) -> str:
        """Generate some text."""
        return f"Write about {topic}"

    result = generate_text("testing")

    assert result == "test response"
    assert len(client.calls) == 1
    assert client.calls[0]["model"] == "gpt-4o-mini"
    assert client.calls[0]["messages"][0]["role"] == "user"
    assert client.calls[0]["messages"][0]["content"] == "Write about testing"


def test_structured_output(mock_client_factory):
    """Test function with structured Pydantic output."""
    client = mock_client_factory('{"summary": "test", "pros": ["a", "b"], "cons": ["c"]}')

    @backend(client, model="gpt-4o-mini", response_format=Summary)
    def summarize(text: str) -> Summary:
        """Summarize text."""
        return f"Summarize: {text}"

    result = summarize("pokemon")

    assert isinstance(result, Summary)
    assert result.summary == "test"
    assert result.pros == ["a", "b"]
    assert result.cons == ["c"]

    # Verify response_format was set
    assert "response_format" in client.calls[0]
    assert client.calls[0]["response_format"]["type"] == "json_schema"
    schema = client.calls[0]["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False


def test_system_prompt(mock_client_factory):
    """Test that system prompt is correctly passed."""
    client = mock_client_factory()

    @backend(client, model="gpt-4o-mini", system="You are helpful")
    def generate(prompt: str) -> str:
        return prompt

    result = generate("test")

    assert len(client.calls[0]["messages"]) == 2
    assert client.calls[0]["messages"][0]["role"] == "system"
    assert client.calls[0]["messages"][0]["content"] == "You are helpful"
    assert client.calls[0]["messages"][1]["role"] == "user"


def test_extra_kwargs(mock_client_factory):
    """Test that extra kwargs are passed to OpenAI API."""
    client = mock_client_factory()

    @backend(client, model="gpt-4o-mini", temperature=0.7, max_tokens=100)
    def generate(prompt: str) -> str:
        return prompt

    result = generate("test")

    assert client.calls[0]["temperature"] == 0.7
    assert client.calls[0]["max_tokens"] == 100


def test_function_must_return_string(mock_client_factory):
    """Test that function must return a string or list."""
    client = mock_client_factory()

    @backend(client, model="gpt-4o-mini")
    def bad_function() -> str:
        return 123  # Not a string!

    with pytest.raises(ValueError, match="must return either a string prompt or a list"):
        bad_function()


def test_run_method(mock_client_factory):
    """Test the run method for non-decorator usage."""
    client = mock_client_factory()

    backend_instance = backend(client, model="gpt-4o-mini")

    def generate(prompt: str) -> str:
        return f"Process: {prompt}"

    result = backend_instance.run(generate, "test")

    assert result == "test response"
    assert client.calls[0]["messages"][0]["content"] == "Process: test"


@pytest.mark.asyncio
async def test_async_basic(async_mock_client_factory):
    """Test async backend basic functionality."""
    client = async_mock_client_factory()

    @async_backend(client, model="gpt-4o-mini")
    def generate_text(topic: str) -> str:
        """Generate text async."""
        return f"Write about {topic}"

    result = await generate_text("testing")

    assert result == "test response"
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_async_structured_output(async_mock_client_factory):
    """Test async backend with structured output."""
    client = async_mock_client_factory('{"summary": "async test", "pros": ["fast"], "cons": []}')

    @async_backend(client, model="gpt-4o-mini", response_format=Summary)
    def summarize(text: str) -> Summary:
        """Summarize async."""
        return f"Summarize: {text}"

    result = await summarize("pokemon")

    assert isinstance(result, Summary)
    assert result.summary == "async test"
    assert result.pros == ["fast"]
    schema = client.calls[0]["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False


@pytest.mark.asyncio
async def test_async_run_method(async_mock_client_factory):
    """Test the async run method."""
    client = async_mock_client_factory()

    backend_instance = async_backend(client, model="gpt-4o-mini")

    def generate(prompt: str) -> str:
        return f"Process: {prompt}"

    result = await backend_instance.run(generate, "test")

    assert result == "test response"


def test_multiple_arguments(mock_client_factory):
    """Test function with multiple arguments."""
    client = mock_client_factory()

    @backend(client, model="gpt-4o-mini")
    def generate(topic: str, style: str, length: int) -> str:
        return f"Write a {length} word {style} piece about {topic}"

    result = generate("AI", "formal", 500)

    content = client.calls[0]["messages"][0]["content"]
    assert "Write a 500 word formal piece about AI" in content


def test_complex_prompt_logic(mock_client_factory):
    """Test that function can have complex prompt generation logic."""
    client = mock_client_factory()

    @backend(client, model="gpt-4o-mini")
    def smart_generate(items: list[str], include_summary: bool) -> str:
        prompt = "Process these items:\n"
        for i, item in enumerate(items, 1):
            prompt += f"{i}. {item}\n"

        if include_summary:
            prompt += "\nProvide a summary at the end."

        return prompt

    result = smart_generate(["apple", "banana"], True)

    content = client.calls[0]["messages"][0]["content"]
    assert "1. apple" in content
    assert "2. banana" in content
    assert "summary" in content
