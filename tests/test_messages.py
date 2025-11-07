import pytest
from pydantic import BaseModel
from smartfunc import backend, async_backend


class Summary(BaseModel):
    """Test model for structured output."""
    summary: str


def test_message_list_basic(mock_client_factory):
    """Test function that returns a list of messages."""
    client = mock_client_factory()

    @backend(client, model="gpt-4o-mini")
    def chat_with_history(user_message: str) -> list:
        """Chat with conversation history."""
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": user_message},
        ]

    result = chat_with_history("What's the weather?")

    assert result == "test response"
    messages = client.calls[0]["messages"]
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[3]["role"] == "user"
    assert messages[3]["content"] == "What's the weather?"


def test_message_list_ignores_system_param(mock_client_factory):
    """Test that system parameter is ignored when messages are provided."""
    client = mock_client_factory()

    @backend(client, model="gpt-4o-mini", system="This should be ignored")
    def chat() -> list:
        """Chat with custom messages."""
        return [
            {"role": "system", "content": "Custom system message"},
            {"role": "user", "content": "Hello"},
        ]

    result = chat()

    messages = client.calls[0]["messages"]
    assert len(messages) == 2
    assert messages[0]["content"] == "Custom system message"


def test_multimodal_content(mock_client_factory):
    """Test function with multimodal content (text + image)."""
    client = mock_client_factory()

    @backend(client, model="gpt-4o-mini")
    def analyze_image(image_base64: str, question: str) -> list:
        """Analyze an image."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            }
        ]

    result = analyze_image("iVBORw0KGgo...", "What's in this image?")

    assert result == "test response"
    messages = client.calls[0]["messages"]
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "What's in this image?"
    assert content[1]["type"] == "image_url"
    assert "data:image/jpeg;base64" in content[1]["image_url"]["url"]


def test_invalid_return_type(mock_client_factory):
    """Test that invalid return types raise errors."""
    client = mock_client_factory()

    @backend(client, model="gpt-4o-mini")
    def bad_function() -> str:
        return {"invalid": "dict"}  # Not string or list

    with pytest.raises(ValueError, match="must return either a string prompt or a list"):
        bad_function()


def test_message_list_with_structured_output(mock_client_factory):
    """Test message list with structured output format."""
    client = mock_client_factory('{"summary": "conversation summary"}')

    @backend(client, model="gpt-4o-mini", response_format=Summary)
    def summarize_conversation() -> list:
        """Summarize a conversation."""
        return [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Summarize our conversation"},
        ]

    result = summarize_conversation()

    assert isinstance(result, Summary)
    assert result.summary == "conversation summary"


@pytest.mark.asyncio
async def test_async_message_list(async_mock_client_factory):
    """Test async backend with message list."""
    client = async_mock_client_factory()

    @async_backend(client, model="gpt-4o-mini")
    def chat() -> list:
        """Async chat."""
        return [
            {"role": "user", "content": "Hello"},
        ]

    result = await chat()

    assert result == "test response"
    messages = client.calls[0]["messages"]
    assert len(messages) == 1
