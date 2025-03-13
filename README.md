<img src="imgs/logo.png" width="125" height="125" align="right" />

### smartfunc

> Turn docstrings into LLM-functions

## Installation

```bash
uv pip install smartfunc
```


## What is this?

Here is a nice example of what is possible with this library:

```python
from smartfunc import backend

@backend("gpt-4")
def generate_summary(text: str):
    """Generate a summary of the following text: {{ text }}"""
    pass
```

The `generate_summary` function will now return a string with the summary of the text that you give it.

## How does it work?

This library wraps around the [llm library](https://llm.datasette.io/en/stable/index.html) made by [Simon Willison](https://simonwillison.net/). The docstring is parsed and turned into a Jinja2 template which we inject with variables to generate a prompt at runtime. We then use the backend given by the decorator to run the prompt and return the result.

The `llm` library is minimalistic and while it does not support all the features out there it does offer a solid foundation to build on. This library is mainly meant as a method to add some syntactic sugar on top. We do get a few nice benefits from the `llm` library though:

- The `llm` library is well maintained and has a large community
- An [ecosystem of backends](https://llm.datasette.io/en/stable/plugins/directory.html) for different LLM providers
- Many of the vendors have `async` support, which allows us to do microbatching
- Many of the vendors have schema support, which allows us to use Pydantic models to define the response
- You can use `.env` files to store your API keys

## Extra features

### Schemas

The following snippet shows how you might create a re-useable backend decorator that uses a system prompt. Also notice how we're able to use a Pydantic model to define the response.

```python
from smartfunc import backend
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(".env")

class Summary(BaseModel):
    summary: str
    pros: list[str]
    cons: list[str]

llmify = backend("gpt-4o-mini", system="You are a helpful assistant.", temperature=0.5)

@llmify
def generate_poke_desc(text: str) -> Summary:
    """Describe the following pokemon: {{ text }}"""
    pass

print(generate_poke_desc("pikachu"))
```

This is the result that we got back:

```python
{
    'summary': 'Pikachu is a small, electric-type Pokémon known for its adorable appearance and strong electrical abilities. It is recognized as the mascot of the Pokémon franchise, with distinctive features and a cheerful personality.', 
    'pros': [
        'Iconic and recognizable worldwide', 
        'Strong electric attacks like Thunderbolt', 
        'Has a cute and friendly appearance', 
        'Evolves into Raichu with a Thunder Stone', 
        'Popular choice in Pokémon merchandise and media'
    ], 
    'cons': [
        'Not very strong in higher-level battles', 
        'Weak against ground-type moves', 
        'Limited to electric-type attacks unless learned through TMs', 
        'Can be overshadowed by other powerful Pokémon in competitive play'
    ],
}
```

Not every backend supports schemas, but you will a helpful error message if that is the case.

> [!NOTE]  
> You might look at this example and wonder if you might be better off using [instructor](https://python.useinstructor.com/). After all, that library has more support for validation of parameters and even has some utilities for multi-turn conversations. And what about [ell](https://github.com/MadcowD/ell) or [marvin](https://www.askmarvin.ai/)?! 
> 
> You will notice that `smartfunc` doesn't do a bunch of things those other libraries do. But the goal here is simplicity and a focus on a specific set of features.  For example; instructor requires you to learn a fair bit more about each individual backend. If you want to to use claude instead of openai then you will need to load in a different library. Similarily I felt that all the other platforms that similar things missing: async support or freedom for vendors. 
>
> The goal here is simplicity during rapid prototyping. You just need to make sure the `llm` plugin is installed and you're good to go. That's it. 


### Async

The library also supports async functions. This is useful if you want to do microbatching or if you want to use the async backends from the `llm` library.

```python
import asyncio
from smartfunc import async_backend
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(".env")


class Summary(BaseModel):
    summary: str
    pros: list[str]
    cons: list[str]


@async_backend("gpt-4o-mini")
async def generate_poke_desc(text: str) -> Summary:
    """Describe the following pokemon: {{ text }}"""
    pass

resp = asyncio.run(generate_poke_desc("pikachu"))
print(resp)
```
