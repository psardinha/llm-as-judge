
from phoenix.client import Client


# -------------------------------------------------------
# 1. Create a labelled dataset in Phoenix
# -------------------------------------------------------

phoenix = Client()

examples = [{"input": {"query": "What is the product return period?"},
             "output": {"relevant_ids": ["D1", "D3"]},
             "metadata": {"query_id": "Q1"}},
            {"input": {"query": "When is payment due?"},
             "output": {"relevant_ids": ["D5"]},
             "metadata": {"query_id": "Q2"}},
            {"input": {"query": "How are refunds processed?"},
             "output": {"relevant_ids": ["D2", "D6"]},
             "metadata": {"query_id": "Q3"}}]

dataset = phoenix.datasets.create_dataset(name="rag-retriever-benchmark",
                                          dataset_description=("Queries and ground-truth relevant chunk ID "
                                                               "for evaluating retrieval quality."),
                                          examples=examples)
print("Created dataset:", dataset)


# -------------------------------------------------------
# 2. Example corpus and mock retriever
#    retrieve() simulates a real retriever.
# -------------------------------------------------------

DOCUMENTS = {"D1": "Eligible products may be returned within 30 days.",
             "D2": "Refunds are issued to the original payment method.",
             "D3": "Returned products must be in original condition.",
             "D4": "The company was founded in 1998.",
             "D5": "Payment is due within 15 days of the invoice.",
             "D6": "Refund processing may take five business days."}

MOCK_RESULTS = {"What is the product return period?": ["D4", "D1", "D3"],
                "When is payment due?": ["D5", "D4", "D2"],
                "How are refunds processed?": ["D2", "D4", "D6"]}

def retrieve(query: str, k: int = 3) -> list[str]:
    """Return ranked document IDs from the retriever."""
    return MOCK_RESULTS[query][:k]

K = 3


# -------------------------------------------------------
# 3. Experiment task: run the retriever
#    Phoenix passes each dataset example's input to this task.
# -------------------------------------------------------

def retrieval_task(input: dict) -> dict:
    "Simulates the action of a retriever"
    query = input["query"]
    ranked_ids = retrieve(query, k=K)

    return {"retrieved_ids": ranked_ids,
            "retrieved_texts": [DOCUMENTS[doc_id] for doc_id in ranked_ids]}


# -------------------------------------------------------
# 4. Calculate information retrieval metrics
# -------------------------------------------------------

def retrieval_metrics(retrieved_ids: list[str],
                      relevant_ids: list[str],
                      k: int = 5) -> dict[str, float]:
    ranked = retrieved_ids[:k]
    relevant = set(relevant_ids)

    # Precision@k: relevant retrieved results / results returned.
    # Use the actual number returned if fewer than k exist.
    precision = sum(doc_id in relevant for doc_id in ranked) / len(ranked) if ranked else 0.0

    # Recall@k: relevant retrieved results / all known relevant IDs.
    recall = len(set(ranked) & relevant) / len(relevant) if relevant else 0.0

    # Reciprocal rank of the first relevant result.
    reciprocal_rank = next((1.0/rank  for rank, doc_id in enumerate(ranked, start=1)  if doc_id in relevant), 0.0)

    return {f"precision_at_{k}": precision,
            f"recall_at_{k}": recall,
            "mrr": reciprocal_rank}


# -------------------------------------------------------
# 5. Phoenix evaluators
# Evaluators receive the task output and reference output.
# -------------------------------------------------------

def make_metric_evaluator(metric_name: str): 
    def evaluator(output: dict, expected: dict) -> dict:
        metrics = retrieval_metrics(retrieved_ids=output["retrieved_ids"], relevant_ids=expected["relevant_ids"], k=K)

        value = metrics[metric_name]

        return {"score": value,
                "label": f"{value:.3f}",
                "explanation": f"{metric_name}={value:.3f}; retrieved={output['retrieved_ids']}; relevant={expected['relevant_ids']}"}

    return evaluator


precision_evaluator = make_metric_evaluator(f"precision_at_{K}")
recall_evaluator = make_metric_evaluator(f"recall_at_{K}")
mrr_evaluator = make_metric_evaluator("mrr")

# -------------------------------------------------------
# 6. Run and store the experiment in Phoenix
# -------------------------------------------------------

experiment = phoenix.experiments.run_experiment(dataset=dataset,
                                                task=retrieval_task,
                                                evaluators={f"precision_at_{K}": precision_evaluator,
                                                            f"recall_at_{K}": recall_evaluator,
                                                            "mrr": mrr_evaluator},
                                                experiment_name="baseline-retriever",
                                                experiment_description="Precision@3, Recall@3 and MRR against labelled chunks.")
print("Experiment completed:", experiment)


# -------------------------------------------------------
# 7. Calculate aggregate metrics across the dataset
# -------------------------------------------------------

per_query_metrics = []

for example in examples:
    query = example["input"]["query"]
    relevant_ids = example["output"]["relevant_ids"]
    retrieved_ids = retrieve(query, k=K)

    scores = retrieval_metrics(retrieved_ids=retrieved_ids, relevant_ids=relevant_ids, k=K)

    per_query_metrics.append({"query": query, "retrieved_ids": retrieved_ids, **scores})

print("\nPer-query metrics:")
for row in per_query_metrics:
    print(row)

print("\nMacro-average metrics:")
for metric_name in [f"precision_at_{K}", f"recall_at_{K}", "mrr"]:
    average = sum(row[metric_name] for row in per_query_metrics) / len(per_query_metrics)
    print(f"{metric_name}: {average:.3f}")
