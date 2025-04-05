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


### Inner function prompt engineering

The simplest way to use `smartfunc` is to just put your prompt in the docstring and to be done with it. You can also run jinja2 in it if you want, but if you need the extra flexibility then you can also use the inner function to write the logic of your promopt. Any string that the inner function returns will be added at the back of the docstring prompt.

```python
import asyncio
from smartfunc import backend
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(".env")


class Summary(BaseModel):
    summary: str
    pros: list[str]
    cons: list[str]

# This would also work, but has the benefit that you can use the inner function to write 
# the logic of your prompt which allows for more flexible prompt engineering
@backend("gpt-4o-mini")
def generate_poke_desc(text: str) -> Summary:
    """Describe the following pokemon: {{ text }}"""
    return " ... but make it sound as if you are a 10 year old child"

resp = generate_poke_desc("pikachu")
print(resp) # This response should now be more child-like
```

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

### Debug mode

The library also supports debug mode. This is useful if you want to see the prompt that was used or if you want to see the response that was returned.

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


@async_backend("gpt-4o-mini", debug=True)
async def generate_poke_desc(text: str) -> Summary:
    """Describe the following pokemon: {{ text }}"""
    pass

resp = asyncio.run(generate_poke_desc("pikachu"))
print(resp)
```

This will return a dictionary with the debug information.

```python
{
    'summary': 'Pikachu is a small, yellow, rodent-like Pokémon known for its electric powers and iconic status as the franchise mascot. It has long ears with black tips, red cheeks that store electricity, and a lightning bolt-shaped tail. Pikachu evolves from Pichu when leveled up with high friendship and can further evolve into Raichu when exposed to a Thunder Stone. Pikachu is often depicted as cheerful, playfully energetic, and is renowned for its ability to generate electricity, which it can unleash in powerful attacks such as Thunderbolt and Volt Tackle.', 
    'pros': [
        'Iconic mascot of the Pokémon franchise', 'Popular among fans of all ages', 'Strong electric-type moves', 'Cute and friendly appearance'
    ], 
    'cons': [
        'Limited range of evolution (only evolves into Raichu)', 'Commonly found, which may reduce uniqueness', 'Vulnerable to ground-type moves', 'Requires high friendship for evolution to Pichu, which can be a long process'
    ], 
    '_debug': {
        'template': 'Describe the following pokemon: {{ text }}', 
        'func_name': 'generate_poke_desc', 
        'prompt': 'Describe the following pokemon: pikachu', 
        'system': None, 
        'template_inputs': {
            'text': 'pikachu'
        }, 
        'backend_kwargs': {}, 
        'datetime': '2025-03-13T16:05:44.754579', 
        'return_type': {
            'properties': {
                'summary': {'title': 'Summary', 'type': 'string'}, 
                'pros': {'items': {'type': 'string'}, 'title': 'Pros', 'type': 'array'}, 
                'cons': {'items': {'type': 'string'}, 'title': 'Cons', 'type': 'array'}
            }, 
            'required': ['summary', 'pros', 'cons'], 
            'title': 'Summary', 
            'type': 'object'
        }
    }
}
```
