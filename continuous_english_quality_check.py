import hashlib
import json
import logging
import pandas as pd
import time
from datetime import datetime, timedelta, timezone

from phoenix.client import Client
from phoenix.evals.llm import LLM
from phoenix.evals import ClassificationEvaluator

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_NAME = "english-quality-tests"

SAMPLE_RATE = 0.50
LOOKBACK_MINUTES = 300
BATCH_LIMIT = 100
ANNOTATION_NAME = "english_quality"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Phoenix client and evaluator
# --------------------------------------------------

client = Client()
llm = LLM(provider="openai", model="gemma4:latest", client="openai", base_url="http://localhost:11434/v1", api_key="ollama")

# Extract the system message from the stored prompt
english_score_template = client.prompts.get(prompt_identifier="english_score_template", tag="v1")

spelling_classifier = ClassificationEvaluator(name=ANNOTATION_NAME,
                                              prompt_template=english_score_template._template["messages"][0]["content"][0]["text"],
                                              llm=llm,
                                              choices={str(i): i for i in range(1, 11)},
                                              direction="maximize")

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def parse_value(value):
    """Decode LLM JSON reply to extract its textual answer."""
    if isinstance(value, str):
        value = json.loads(value)
    if (not isinstance(value, dict)) or (not isinstance(value.get("choices"), list)):
        return None
    for message in value["choices"]:
        if message.get("message") and message["message"].get("content"):
            return value["choices"][0]["message"]["content"]
    return None


def get_answer(output_value):
    """Extract the assistant's answer from a span output."""
    output_value = parse_value(output_value)
    if isinstance(output_value, dict):
        # Adapt this if your application uses a different output key.
        for key in ("answer", "response", "output", "text"):
            if isinstance(output_value.get(key), str):
                return output_value[key]
    if isinstance(output_value, str):
        return output_value
    return None


def selected_for_sampling(trace_id):
    """Deterministic sampling: sampling of spans is determined by the traceID of the trace it is assigned to.
    Thus, the whole set of spands of each trace are either simultaneously all sampled or simultaneously not sampled."""
    digest = hashlib.sha256(str(trace_id).encode("utf-8")).hexdigest()
    fraction = int(digest[:16], 16) / (16**16)
    return fraction < SAMPLE_RATE


def already_annotated(span_ids):
    """Find spans that already have <ANNOTATION_NAME> evaluation annotation."""
    if not span_ids:
        return set()
    annotations_df = client.spans.get_span_annotations_dataframe(span_ids=span_ids,
                                                                 project_identifier=PROJECT_NAME,
                                                                 include_annotation_names=[ANNOTATION_NAME])
    if annotations_df.empty:
        return set()

    # Phoenix annotation dataframes commonly expose span_id as a column.
    if annotations_df.index.name == "span_id":
        return set(annotations_df.index.dropna().astype(str))
    elif "span_id" in annotations_df.columns:
        return set(annotations_df["span_id"].dropna().astype(str))
    return set()


# --------------------------------------------------
# Process previously recorded spans
# --------------------------------------------------

def evaluate_recorded_traces():
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=LOOKBACK_MINUTES)

    spans_df = client.spans.get_spans_dataframe(project_identifier=PROJECT_NAME,
                                                start_time=start_time,
                                                end_time=end_time,
                                                root_spans_only=False,
                                                limit=BATCH_LIMIT)

    kind_column = "span_kind"
    if kind_column in spans_df.columns:
        spans_df = spans_df[spans_df[kind_column].astype(str).str.upper() == "LLM"]
    else:
        print(f"LLM span-kind attribute not found. Available columns: {spans_df.columns.tolist()}")
        return 0
    if spans_df.empty:
        logger.info("No relevant spans of kind LLM found in the time window.")
        return 0

    # In Phoenix DataFrames, the span ID is often the index.
    if spans_df.index.name == "context.span_id" and "context.span_id" not in spans_df.columns:
        spans_df = spans_df.reset_index()

    if "context.span_id" in spans_df.columns:
        spans_df["span_id"] = spans_df["context.span_id"].astype(str)
    elif "span_id" not in spans_df.columns:
        raise ValueError(f"Cannot find span IDs. Columns: {list(spans_df.columns)}")

    if "context.trace_id" not in spans_df.columns:
        raise ValueError("Cannot find context.trace_id; cannot sample by trace.")

    output_column = "attributes.llm.output_messages"
    output_column = "attributes.output.value"
    if output_column not in spans_df.columns:
        raise ValueError(f"Cannot find {output_column}. Available columns: {list(spans_df.columns)}")
    # Check annotations before calling the evaluator.
    existing_ids = already_annotated(spans_df["span_id"].dropna().astype(str).tolist())
    annotations_to_save = []

    for _, span in spans_df.iterrows():
        span_id = str(span["span_id"])
        trace_id = span["context.trace_id"]
        print(f"Cycle for trace/span: {trace_id}/{span_id}")

        if span_id in existing_ids:
            logger.info("Skipping span %s: already %s-annotated", span_id, ANNOTATION_NAME)
            continue

        if not selected_for_sampling(trace_id):
            print(f"trace_id {trace_id} skipped")
            continue

        answer = get_answer(span[output_column])

        if not answer or not answer.strip():
            logger.warning("Skipping span %s: no text output", span_id)
            continue

        try:
            result = spelling_classifier.evaluate({"context": answer})[0]
            print(f"Text: {answer}; result: {result}")
            annotations_to_save.append({"span_id": span_id,
                                        "score": float(result.score),
                                        "label": str(result.label),
                                        "explanation": str(result.explanation)})
            logger.info("Evaluated span %s: score=%s", span_id, result.score)
        except Exception:
            logger.exception("Evaluation failed for span %s", span_id)

    if not annotations_to_save:
        logger.info("No new annotations to save.")
        return 0

    # Save the results against their corresponding Phoenix spans.
    annotations_df = pd.DataFrame(annotations_to_save)

    client.spans.log_span_annotations_dataframe(dataframe=annotations_df,
                                                annotation_name=ANNOTATION_NAME,
                                                annotator_kind="LLM",
                                                sync=True)

    logger.info("Saved %d annotations.", len(annotations_to_save))
    return len(annotations_to_save)


# --------------------------------------------------
# Periodic polling loop
# --------------------------------------------------

if __name__ == "__main__":
    while True:
        try:
            evaluate_recorded_traces()
        except Exception as e:
            logger.exception(f"Trace evaluation job failed: {e}")

        # Run again every five minutes.
        time.sleep(LOOKBACK_MINUTES)