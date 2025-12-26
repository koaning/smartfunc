# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "marimo",
#   "openai",
#   "python-dotenv",
#   "pydantic",
# ]
# ///
import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from dotenv import load_dotenv
    from openai import OpenAI
    from smartfunc import learnable, Pipeline

    load_dotenv(".env", override=True)
    return OpenAI, Pipeline, learnable, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Learnable Pipelines (Prompt Optimization)

    This notebook shows how to define prompt-based learnable functions and
    optimize them against input/output examples with a metric.

    We'll train a simple classifier to detect whether a sentence about
    \"Python\" is about programming or the animal.
    """)
    return


@app.cell
def _(OpenAI):
    client = OpenAI()
    return (client,)


@app.cell
def _(Pipeline, client, learnable):
    @learnable(
        client,
        model="gpt-4o-mini",
        prompt=(
            "Classify if the sentence is about programming (Python language) "
            "or about the animal. Reply with exactly one word: PROGRAMMING or ANIMAL.\n"
            "Sentence: {text}"
        ),
    )
    def classify(text: str):
        pass  # auto-maps inputs into the prompt

    pipeline = Pipeline(classify)
    return (pipeline,)


@app.cell
def _():
    examples = [
        {"input": "I wrote a Python script to automate reports.", "output": "PROGRAMMING"},
        {"input": "The python curled up on the warm rock.", "output": "ANIMAL"},
        {"input": "We deployed a Python service to production.", "output": "PROGRAMMING"},
        {"input": "The python shed its skin overnight.", "output": "ANIMAL"},
        {"input": "Python has great libraries for data analysis.", "output": "PROGRAMMING"},
        {"input": "A python can swallow prey whole.", "output": "ANIMAL"},
    ]

    def metric(example: dict) -> float:
        return 1.0 if example["prediction"] == example["output"] else 0.0
    return examples, metric


@app.cell
def _(examples, metric, pipeline):
    run_learning = False

    if run_learning:
        pipeline.learn(examples=examples, metric=metric, steps=3, candidates=2)
    return


@app.cell
def _(pipeline):
    run_inference = False

    if run_inference:
        result = pipeline("Python lets you build quick prototypes.")
        result
    return


@app.cell
def _(pipeline):
    current_prompt = pipeline.classify.prompt
    original_prompt = pipeline.classify.base_prompt
    return


if __name__ == "__main__":
    app.run()
