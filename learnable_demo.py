import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from dotenv import load_dotenv
    from openai import OpenAI
    from smartfunc import learnable, Pipeline

    load_dotenv(".env", override=True)
    return Pipeline, OpenAI, learnable, load_dotenv, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Learnable Pipelines (Prompt Optimization)

        This notebook shows how to define prompt-based learnable functions and
        optimize them against input/output examples with a metric.
        """
    )
    return


@app.cell
def _(OpenAI):
    client = OpenAI()
    return (client,)


@app.cell
def _(Pipeline, client, learnable):
    @learnable(client, model="gpt-4o-mini", prompt="Filter: {x}")
    def filter_func(x: str):
        pass  # auto-maps inputs into the prompt

    @learnable(client, model="gpt-4o-mini", prompt="Rank: {x}")
    def ranker_func(x: str):
        pass

    pipeline = Pipeline(filter_func, ranker_func)
    return filter_func, pipeline, ranker_func


@app.cell
def _():
    examples = [
        {"input": "cats", "output": "ranked cats"},
        {"input": "dogs", "output": "ranked dogs"},
    ]

    def metric(example: dict) -> float:
        return 1.0 if example["prediction"] == example["output"] else 0.0

    return examples, metric


@app.cell
def _(examples, metric, pipeline):
    run_learning = False

    if run_learning:
        pipeline.learn(examples=examples, metric=metric, steps=3, candidates=2)

    return (run_learning,)


@app.cell
def _(pipeline):
    run_inference = False

    if run_inference:
        result = pipeline("cats")
        result
    return (run_inference,)


@app.cell
def _(pipeline):
    current_prompt = pipeline.filter_func.prompt
    original_prompt = pipeline.filter_func.base_prompt
    return current_prompt, original_prompt


if __name__ == "__main__":
    app.run()
