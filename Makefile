install: 
	python -m pip install uv
	uv venv
	uv pip install -e .
	uv pip install pytest pytest-asyncio llm-markov

pypi:
	uv build
	uv publish

check:
	uv run pytest tests

clean:
	rm -rf __pycache__ .pytest_cache dist
