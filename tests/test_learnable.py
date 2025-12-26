import pytest

from smartfunc import learnable, async_learnable, Pipeline


class QueueMessage:
    def __init__(self, content: str):
        self.content = content


class QueueChoice:
    def __init__(self, content: str):
        self.message = QueueMessage(content)


class QueueCompletion:
    def __init__(self, content: str):
        self.choices = [QueueChoice(content)]


class QueueCompletions:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("No queued responses left for mock client.")
        return QueueCompletion(self.responses.pop(0))


class QueueChat:
    def __init__(self, responses: list[str]):
        self.completions = QueueCompletions(responses)


class QueueOpenAI:
    def __init__(self, responses: list[str]):
        self.chat = QueueChat(responses)

    @property
    def calls(self):
        return self.chat.completions.calls


def test_learnable_auto_map(mock_client_factory):
    client = mock_client_factory()

    @learnable(client, model="gpt-4o-mini", prompt="Hello {name}")
    def greet(name: str):
        pass

    result = greet("Vincent")

    assert result == "test response"
    assert client.calls[0]["messages"][0]["content"] == "Hello Vincent"
    assert greet.base_prompt == "Hello {name}"
    assert greet.prompt == "Hello {name}"


def test_learnable_dict_return(mock_client_factory):
    client = mock_client_factory()

    @learnable(client, model="gpt-4o-mini", prompt="X {x} Y {y}")
    def combine(x: str, y: str):
        return {"x": x, "y": y}

    result = combine("a", "b")

    assert result == "test response"
    assert client.calls[0]["messages"][0]["content"] == "X a Y b"


def test_pipeline_chain(mock_client_factory):
    client = mock_client_factory()

    @learnable(client, model="gpt-4o-mini", prompt="Filter {x}")
    def filter_func(x: str):
        pass

    @learnable(client, model="gpt-4o-mini", prompt="Rank {x}")
    def ranker_func(x: str):
        pass

    pipeline = Pipeline(filter_func, ranker_func)
    result = pipeline("hello")

    assert result == "test response"
    assert len(client.calls) == 2
    assert pipeline.filter_func is filter_func
    assert pipeline.ranker_func is ranker_func


def test_pipeline_learn_updates_prompt():
    client = QueueOpenAI(["bad", "better prompt", "good"])

    @learnable(client, model="gpt-4o-mini", prompt="Initial {x}")
    def predict(x: str):
        pass

    pipeline = Pipeline(predict)
    examples = [{"input": "hi", "output": "good"}]

    def metric(example: dict) -> float:
        return 1.0 if example["prediction"] == example["output"] else 0.0

    pipeline.learn(examples=examples, metric=metric, steps=1, candidates=1)

    assert predict.prompt == "better prompt"


def test_pipeline_learn_requires_metric(mock_client_factory):
    client = mock_client_factory()

    @learnable(client, model="gpt-4o-mini", prompt="Hello {name}")
    def greet(name: str):
        pass

    pipeline = Pipeline(greet)
    with pytest.raises(ValueError, match="metric must be provided"):
        pipeline.learn(examples=[{"input": "x", "output": "y"}], metric=None)


@pytest.mark.asyncio
async def test_async_learnable_basic(async_mock_client_factory):
    client = async_mock_client_factory()

    @async_learnable(client, model="gpt-4o-mini", prompt="Hello {name}")
    async def greet(name: str):
        return None

    result = await greet("Vincent")

    assert result == "test response"
    assert client.calls[0]["messages"][0]["content"] == "Hello Vincent"
