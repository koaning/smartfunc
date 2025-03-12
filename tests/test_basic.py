import pytest 
from smartfunc import backend


@pytest.mark.parametrize("text", ["Hello, world!", "Hello, programmer!"])
def test_basic(text):
    @backend("markov", delay=0, length=10)
    def generate_summary(t):
        """Generate a summary of the following text: {{ t }}"""
        pass

    assert text in generate_summary(text) 
