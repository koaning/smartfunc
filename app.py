import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import requests as rq
    from dotenv import load_dotenv

    load_dotenv(".env", override=True)
    return load_dotenv, mo, rq


@app.cell(hide_code=True)
def _(rq):
    url = "https://raw.githubusercontent.com/chalda-pnuzig/emojis.json/refs/heads/master/src/list.json"

    emoji = rq.get(url).json()['emojis']
    return emoji, url


@app.cell(hide_code=True)
def _():
    from instructor.batch import BatchJob
    from pydantic import BaseModel, Field
    from typing import Literal

    class EmojiDescription(BaseModel):
        terms: list[str] = Field(..., description="List of words/phrases that could fit the emoji. List can be long, around 10 examples.")
        description: str = Field(..., description="Describes the emoji at length. It does not describe what it looks like, but rather what the symbol could mean and what it is typically used for. Two, max three, sentences.")
    return BaseModel, BatchJob, EmojiDescription, Field, Literal


@app.cell
def _(EmojiDescription, cache, json, llm):
    model = llm.get_async_model("claude-3.5-haiku")

    async def get_info(e):
        resp = await model.prompt(f"What can you tell me about this emoji: {e['emoji']}", schema=EmojiDescription, log=True)
        cache[e['emoji']] = {**e, 'response': json.loads(await resp.text())}
        return cache[e['emoji']]
    return get_info, model


@app.cell
def _():
    from diskcache import Cache

    cache = Cache("emojidb")
    return Cache, cache


@app.cell
def _(cache, emoji):
    todo = [e for e in emoji if e['emoji'] not in cache][:250]
    len(todo), len(cache), len(emoji)
    return (todo,)


@app.cell
def _():
    # results = await async_map_with_retry(
    #     items=todo,
    #     func=get_info,
    #     max_concurrency=5,
    #     max_retries=3,
    #     show_progress=True
    # )
    return


@app.cell
def _(cache):
    import polars as pl 
    from lazylines import LazyLines

    pl.DataFrame(
        LazyLines([cache[k] for k in cache.iterkeys()])
          .mutate(
              desc=lambda d: d['response']['description'],
              terms=lambda d: d['response']['terms']
          )
          .drop("response")
          .show()
    )
    return LazyLines, pl


@app.cell(hide_code=True)
def _(mo):
    import asyncio
    import random
    import logging
    import tqdm
    from typing import List, Dict, Any, Callable, Optional

    async def process_with_retry(
        func, 
        item, 
        max_retries=3, 
        initial_backoff=1.0, 
        backoff_factor=2.0, 
        jitter=0.1,
        timeout=None,
        on_success=None,
        on_failure=None,
        logger=None
    ):
        """Process a single item with retry logic and backoff."""
        logger = logger or logging.getLogger(__name__)
        attempts = 0
        last_exception = None

        while attempts <= max_retries:
            try:
                # Add timeout if specified
                if timeout is not None:
                    result = await asyncio.wait_for(func(item), timeout=timeout)
                else:
                    result = await func(item)

                # Call success callback if provided
                if on_success:
                    on_success(item, result)

                return item, result, None

            except Exception as e:
                attempts += 1
                last_exception = e

                if attempts <= max_retries:
                    # Calculate backoff time with jitter
                    backoff_time = initial_backoff * (backoff_factor ** (attempts - 1))
                    jitter_amount = backoff_time * jitter
                    actual_backoff = backoff_time + random.uniform(-jitter_amount, jitter_amount)
                    actual_backoff = max(0.1, actual_backoff)  # Ensure minimum backoff

                    logger.warning(
                        f"Attempt {attempts}/{max_retries} failed for item {item}. "
                        f"Retrying in {actual_backoff:.2f}s. Error: {str(e)}"
                    )

                    await asyncio.sleep(actual_backoff)
                else:
                    if on_failure:
                        on_failure(item, last_exception)

                    logger.error(
                        f"All {max_retries} retry attempts failed for item {item}. "
                        f"Final error: {str(last_exception)}"
                    )

        return item, None, last_exception

    async def async_map_worker(
        items, 
        func, 
        semaphore,
        max_retries=3, 
        initial_backoff=1.0, 
        backoff_factor=2.0, 
        jitter=0.1,
        timeout=None,
        on_success=None,
        on_failure=None,
        logger=None
    ):
        """Map an async function over items with concurrency control."""
        async def bounded_process(item):
            async with semaphore:
                return await process_with_retry(
                    func, 
                    item, 
                    max_retries, 
                    initial_backoff, 
                    backoff_factor, 
                    jitter,
                    timeout,
                    on_success,
                    on_failure,
                    logger
                )

        # Create tasks
        tasks = [bounded_process(item) for item in items]
        return tasks

    def async_map_with_retry(
        items: List[Dict[Any, Any]],
        func: Callable,
        max_concurrency: int = 10,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        backoff_factor: float = 2.0,
        jitter: float = 0.1,
        timeout: Optional[float] = None,
        on_success: Optional[Callable] = None,
        on_failure: Optional[Callable] = None,
        show_progress: bool = True,
        description: str = "Processing items",
        logger: Optional[logging.Logger] = None
    ):
        """
        Map an async function over a list of dictionaries with progress tracking and retry.

        Args:
            items: List of dictionaries to process
            func: Async function that takes a dictionary and returns a result
            max_concurrency: Maximum number of concurrent tasks
            max_retries: Maximum number of retry attempts
            initial_backoff: Initial backoff time in seconds
            backoff_factor: Multiplier for backoff on successive retries
            jitter: Random jitter factor to avoid thundering herd
            timeout: Maximum time to wait for a task to complete (None = wait forever)
            on_success: Callback function to run on successful processing
            on_failure: Callback function to run when an item fails after all retries
            show_progress: Whether to show progress bar
            description: Description for progress bar
            logger: Optional logger for detailed logging

        Returns:
            List of tuples (original_dict, result_or_None, exception_or_None)
        """
        logger = logger or logging.getLogger(__name__)

        async def main():
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(max_concurrency)

            # Get tasks
            tasks = await async_map_worker(
                items, 
                func, 
                semaphore,
                max_retries, 
                initial_backoff, 
                backoff_factor, 
                jitter,
                timeout,
                on_success,
                on_failure,
                logger
            )

            # Set up progress bar if requested
            if show_progress:
                results = []
                with mo.status.progress_bar(total=len(tasks), title=description) as progress_bar:
                    for task in asyncio.as_completed(tasks):
                        result = await task
                        results.append(result)
                        progress_bar.update()
                return results
            else:
                # Without progress bar, just gather all results
                return await asyncio.gather(*tasks)

        return main()
    return (
        Any,
        Callable,
        Dict,
        List,
        Optional,
        async_map_with_retry,
        async_map_worker,
        asyncio,
        logging,
        process_with_retry,
        random,
        tqdm,
    )


@app.cell
def _():
    import inspect
    import json 
    import llm
    from typing import TypeVar, get_type_hints
    from functools import wraps
    from jinja2 import Template
    from diskcache import Cache
    return Cache, Template, TypeVar, get_type_hints, inspect, json, llm, wraps


@app.cell
def _(
    BaseModel,
    Cache,
    Callable,
    Template,
    get_type_hints,
    inspect,
    json,
    llm,
    wraps,
):
    class backend:
        def __init__(self, name, system=None, cache=None, **kwargs):
            self.model = llm.get_model(name)
            self.system = system
            self.kwargs = kwargs
            self.cache = Cache(cache) if isinstance(cache, str) else cache

        def __call__(self, func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                signature = inspect.signature(func)
                docstring = inspect.getdoc(func) or ""
                type_hints = get_type_hints(func)

                # We only support Pydantic now
                if type_hints.get('return', None):
                    assert issubclass(type_hints.get('return', None), BaseModel), "Output type must be Pydantic class"
            
                # Create a dictionary of parameter types
                param_types = {name: param.default for name, param in signature.parameters.items()}
                bound_args = signature.bind(*args, **kwargs)
                bound_args.apply_defaults()  # Apply default values for missing parameters
                all_kwargs = bound_args.arguments
            
                template = Template(docstring)
                formatted_docstring = template.render(**all_kwargs)
                cache_key = docstring + json.dumps(all_kwargs) + str(type_hints.get('return', None))
            
                if self.cache:
                    if cache_key in self.cache:
                        return self.cache[cache_key]
            
                # Call the prompt, with schema if given
                resp = self.model.prompt(
                    formatted_docstring, 
                    system=self.system,
                    schema=type_hints.get('return', None),
                    **kwargs
                )
                if type_hints.get('return', None):
                    out = json.loads(resp.text())
                out = resp.text()

                if self.cache:
                    self.cache[cache_key] = out
                return out

            return wrapper

        def run(self, func, *args, **kwargs):
            new_func = self(func)
            return new_func(*args, **kwargs)
    return (backend,)


@app.cell
def _(
    BaseModel,
    Callable,
    Template,
    get_type_hints,
    inspect,
    json,
    llm,
    wraps,
):
    class async_backend:
        def __init__(self, name, system=None, **kwargs):
            self.model = llm.get_async_model(name)
            self.system = system
            self.kwargs = kwargs

        def __call__(self, func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                signature = inspect.signature(func)
                docstring = inspect.getdoc(func) or ""
                type_hints = get_type_hints(func)

                # We only support Pydantic now
                if type_hints.get('return', None):
                    assert issubclass(type_hints.get('return', None), BaseModel), "Output type must be Pydantic class"
            
                # Create a dictionary of parameter types
                param_types = {name: param.default for name, param in signature.parameters.items()}
                bound_args = signature.bind(*args, **kwargs)
                bound_args.apply_defaults()  # Apply default values for missing parameters
                all_kwargs = bound_args.arguments
            
                template = Template(docstring)
                formatted_docstring = template.render(**all_kwargs)
            
                # Call the prompt, with schema if given
                resp = await self.model.prompt(
                    formatted_docstring, 
                    system=self.system,
                    schema=type_hints.get('return', None),
                    **kwargs
                )
                text = await resp.text()
                if type_hints.get('return', None):
                    return json.loads(text)
                return text

            return wrapper

        async def run(self, func, *args, **kwargs):
            new_func = self(func)
            return new_func(*args, **kwargs)
    return (async_backend,)


@app.cell
def _(BaseModel, async_backend):
    class Out(BaseModel):
        result: int
    
    @async_backend("claude-3.5-haiku")
    def foobar(a, b) -> Out:
        """
        {{a}} + {{b}} =
        """
    return Out, foobar


@app.cell
async def _(foobar):
    await foobar(1, 2)
    return


@app.cell
def _():
    # @backend("claude-3.5-haiku")
    # async def _foobar(a, b) -> Out:
    #     """
    #     {{a}} + {{b}} =
    #     """

    # _foobar(1, 2)
    return


if __name__ == "__main__":
    app.run()
