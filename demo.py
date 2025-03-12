from smartfunc import backend
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(".env")

class Summary(BaseModel):
    summary: str
    pros: list[str]
    cons: list[str]

llmify = backend("gpt-4o-mini", system="You are a pirate.", temperature=0.5)

@llmify
def generate_poke_desc(text: str) -> Summary:
    """Describe the following pokemon: {{ text }}"""
    pass

print(generate_poke_desc("pikachu"))