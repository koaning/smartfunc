from pydantic import BaseModel
import pytest 

from smartfunc import backend, async_backend


@pytest.mark.parametrize("text", ["Hello, world!", "Hello, programmer!"])
def test_basic(text):
    @backend("markov")
    def generate_summary(t):
        """Generate a summary of the following text: {{ t }}"""
        pass

    assert text in generate_summary(text) 


def test_schema_error():
    with pytest.raises(ValueError):
        class OutputModel(BaseModel):
            result: str

        @backend("markov", delay=0, length=10)
        def generate_summary(t) -> OutputModel:
            """Generate a summary of the following text: {{ t }}"""
            pass

        generate_summary("Hello, world!")
