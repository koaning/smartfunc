import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Structured Output
    """)
    return


@app.cell
def _():
    from smartfunc import backend
    from openai import OpenAI
    from dotenv import load_dotenv
    from pydantic import BaseModel 

    load_dotenv(".env")

    client = OpenAI()

    class Summary(BaseModel):
        summary: str
        pros: list[str]
        cons: list[str]
    return Summary, backend, client


@app.cell
def _(Summary, backend, client):
    @backend(client, model="gpt-4o-mini", response_format=Summary)
    def analyze_pokemon(name: str) -> str:
        return f"Describe the following pokemon: {name}"
    return (analyze_pokemon,)


@app.cell
def _(analyze_pokemon):
    result = analyze_pokemon("pikachu")
    print(result.summary)
    print(result.pros)
    print(result.cons)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Images

    You can also return conversations, which let's you re-use OpenAI's SDK for messages.
    """)
    return


@app.cell
def _():
    url = "https://c02.purpledshub.com/uploads/sites/41/2023/01/How-to-see-the-Wolf-Moon-in-2023--4bb6bb7.jpg?w=940&webp=1"
    return (url,)


@app.cell
def _(backend, client):
    @backend(client, model="gpt-4o-mini")
    def desc_image(url: str) -> str:
        return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the following image:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": url},
                        },
                    ],
                }
            ]
    return (desc_image,)


@app.cell
def _(desc_image, url):
    desc_image(url)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
