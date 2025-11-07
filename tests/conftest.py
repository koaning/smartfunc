"""Pytest fixtures and mock clients for smartfunc tests."""

import pytest
from typing import Optional


class MockMessage:
    """Mock OpenAI message object."""

    def __init__(self, content: str):
        self.content = content


class MockChoice:
    """Mock OpenAI choice object."""

    def __init__(self, content: str):
        self.message = MockMessage(content)


class MockCompletion:
    """Mock OpenAI completion response."""

    def __init__(self, content: str):
        self.choices = [MockChoice(content)]


class MockCompletions:
    """Mock OpenAI completions interface."""

    def __init__(self, response_content: str):
        self.response_content = response_content
        self.calls = []

    def create(self, **kwargs):
        """Mock the create method and store call arguments."""
        self.calls.append(kwargs)
        return MockCompletion(self.response_content)


class MockChat:
    """Mock OpenAI chat interface."""

    def __init__(self, response_content: str):
        self.completions = MockCompletions(response_content)


class MockOpenAI:
    """Duck-typed mock OpenAI client for testing.

    This provides just enough interface to work with smartfunc's backend
    decorators while allowing inspection of calls made during tests.

    Args:
        response_content: The content string to return from API calls

    Example:
        >>> client = MockOpenAI(response_content='{"result": "test"}')
        >>> # Use with backend decorator
        >>> # Later, inspect calls:
        >>> client.chat.completions.calls[0]["model"]
        'gpt-4o-mini'
    """

    def __init__(self, response_content: str = "test response"):
        self.response_content = response_content
        self.chat = MockChat(response_content)

    @property
    def calls(self):
        """Convenience property to access calls made to completions.create()."""
        return self.chat.completions.calls


class MockAsyncCompletions:
    """Mock async OpenAI completions interface."""

    def __init__(self, response_content: str):
        self.response_content = response_content
        self.calls = []

    async def create(self, **kwargs):
        """Mock the async create method and store call arguments."""
        self.calls.append(kwargs)
        return MockCompletion(self.response_content)


class MockAsyncChat:
    """Mock async OpenAI chat interface."""

    def __init__(self, response_content: str):
        self.completions = MockAsyncCompletions(response_content)


class MockAsyncOpenAI:
    """Duck-typed mock AsyncOpenAI client for testing.

    Async version of MockOpenAI.

    Args:
        response_content: The content string to return from API calls
    """

    def __init__(self, response_content: str = "test response"):
        self.response_content = response_content
        self.chat = MockAsyncChat(response_content)

    @property
    def calls(self):
        """Convenience property to access calls made to completions.create()."""
        return self.chat.completions.calls


@pytest.fixture
def mock_client_factory():
    """Factory to create mock OpenAI clients with custom responses.

    Returns a function that creates MockOpenAI instances with the
    specified response content.

    Usage:
        def test_something(mock_client_factory):
            client = mock_client_factory("custom response")
            # or for JSON:
            client = mock_client_factory('{"key": "value"}')
    """
    def _create(response_content: str = "test response"):
        return MockOpenAI(response_content=response_content)
    return _create


@pytest.fixture
def async_mock_client_factory():
    """Factory to create async mock OpenAI clients with custom responses.

    Returns a function that creates MockAsyncOpenAI instances with the
    specified response content.

    Usage:
        async def test_something(async_mock_client_factory):
            client = async_mock_client_factory("custom response")
    """
    def _create(response_content: str = "test response"):
        return MockAsyncOpenAI(response_content=response_content)
    return _create
