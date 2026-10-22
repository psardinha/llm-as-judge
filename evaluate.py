"""Benchmark and evaluate the customer-support agent with Phoenix."""

import json
import argparse
from openai import OpenAI
from phoenix.client import Client
from phoenix.evals.llm import LLM
from phoenix.evals.metrics import CorrectnessEvaluator, CompletenessEvaluator
from phoenix.evals import ClassificationEvaluator
from opentelemetry import trace

from agent import run_agent, tracer_provider


DATASET_NAME = "simple-tool-agent-evaluation"

# ---------------------------------------------------------
# 1. Configure Phoenix and the judge client
# ---------------------------------------------------------

phoenix = Client()
judge_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
tracer = trace.get_tracer("simple-tool-agent-evaluation")


# ---------------------------------------------------------
# 2. Deterministic evaluator: was the tool called as expected?
# ---------------------------------------------------------

def tool_call_check(input, output, expected):
    should_call = expected["tool_should_be_called"]
    was_called = output["tool_called"]
    passed = was_called == should_call
    return {"score": float(passed),
            "label": "pass" if passed else "fail",
            "explanation": f"Expected tool call: {should_call}; Actual tool call: {was_called}"}


# ---------------------------------------------------------
# 3. LLM-as-a-judge: assess the final answer
# ---------------------------------------------------------

def judge_answer(question, reference, answer):
    response = judge_client.chat.completions.create(
        model="gemma4:latest",
        temperature=0,
        messages=[{"role": "system",
                   "content": """
You are an impartial evaluator.

Assess the candidate answer for factual correctness, relevance, and completeness compared with the reference.

Return JSON only:
{
  "score": 0 or 1,
  "reason": "brief explanation"
}

Use score 1 only if the answer is correct and sufficiently complete. Otherwise use 0.
""",
                  },
                  {"role": "user",
                   "content": f"""
Question: {question}

Reference answer: {reference}

Candidate answer: {answer}
""",
                  }],
        response_format={"type": "json_object"})

    result = json.loads(response.choices[0].message.content)
    if result.get("score") not in (0, 1):
        raise ValueError(f"Invalid judge score: {result}")
    return result


def answer_quality(input, output, expected):
    result = judge_answer(question=input["question"],
                          reference=expected["expected_answer"],
                          answer=output["answer"])
    return {"score": float(result["score"]),
            "label": "good" if result["score"] == 1 else "poor",
            "explanation": result["reason"]}


# ---------------------------------------------------------
# 4. English language quality evaluator
# ---------------------------------------------------------

llm = LLM(provider="openai",
          model="gemma4:latest",
          client="openai",
          base_url="http://localhost:11434/v1",
          api_key="ollama")

english_score_template = phoenix.prompts.get(prompt_identifier="english_score_template", tag="v1")

spelling_classifier = ClassificationEvaluator(name="language checker",
                                              prompt_template=english_score_template._template["messages"][0]["content"][0]["text"],
                                              llm=llm,
                                              choices={str(i): i for i in range(1, 11)},
                                              direction="maximize")


def english_quality_reply_eval(input, output, expected):
    # Prefix an intentionally flawed sentence to demonstrate detection.
    result = spelling_classifier.evaluate({"context": "Ear you goes:\n" + output["answer"]})[0]
    return {"score": float(result.score),
            "label": result.label,
            "explanation": result.explanation}


# ---------------------------------------------------------
# 5. Phoenix correctness and completeness evaluators
# ---------------------------------------------------------

correctness_eval = CorrectnessEvaluator(llm=llm)
completeness_eval = CompletenessEvaluator(llm=llm)


def phoenix_completeness(input, output, expected):
    conversation = f"User: {input['question']}\nAssistant: {output['answer']}"
    result = completeness_eval.evaluate({"conversation": conversation})[0]
    return {"score": float(result.score),
            "label": result.label,
            "explanation": result.explanation}


def phoenix_correctness(input, output, expected):
    result = correctness_eval.evaluate({"input": input["question"], "output": output["answer"]})[0]
    return {"score": float(result.score),
            "label": result.label,
            "explanation": result.explanation}


# ---------------------------------------------------------
# 6. Run the benchmark experiment
# ---------------------------------------------------------

def task(input):
    with tracer.start_as_current_span("run_agent"):
        return run_agent(input["question"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the customer-support benchmark dataset.")
    parser.add_argument("dataset_name", nargs="?", default=DATASET_NAME, help=f"Dataset name (default: {DATASET_NAME})")
    args = parser.parse_args()
    dataset = phoenix.datasets.get_dataset(dataset=args.dataset_name)
    experiment = phoenix.experiments.run_experiment(dataset=dataset,
                                                    task=task,
                                                    evaluators=[tool_call_check, answer_quality, phoenix_correctness,
                                                                phoenix_completeness, english_quality_reply_eval],
                                                    experiment_name="simple-tool-agent-baseline")

    print("Experiment completed.")
    print("Inspect the dataset's Experiments tab in Phoenix.")
    tracer_provider.force_flush()
