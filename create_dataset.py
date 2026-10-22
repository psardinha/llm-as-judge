"""Create the customer-support benchmark dataset."""

import argparse
from phoenix.client import Client

DATASET_NAME = "simple-tool-agent-evaluation"

examples = [{"input": {"question": "Where is order 1001, and when will it arrive?"},
             "output": {"expected_answer": "Order 1001 has shipped and is expected to arrive tomorrow.",
                        "tool_should_be_called": True},
            },
            {"input": {"question": "What is the status of order 1002?"},
             "output": {"expected_answer": "Order 1002 is still being processed.",
                        "tool_should_be_called": True},
            },
            {"input": {"question": "What can you help me with?"},
             "output": {"expected_answer": "I can help check an order's status if you provide an order number.",
                        "tool_should_be_called": False}
            }]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the customer-support benchmark dataset.")
    parser.add_argument("dataset_name", nargs="?", default=DATASET_NAME, help=f"Dataset name (default: {DATASET_NAME})")
    args = parser.parse_args()
    phoenix = Client()

    dataset = phoenix.datasets.create_dataset(name=args.dataset_name, examples=examples)
    print(f"Created dataset: {DATASET_NAME}")
