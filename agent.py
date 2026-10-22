"""Customer-support agent with a simulated order-status tool."""

import json

from openai import OpenAI
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace


# ---------------------------------------------------------
# 1. Configure Phoenix tracing and the Ollama client
# ---------------------------------------------------------

openai_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

tracer_provider = register(project_name="simple-tool-agent")
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = trace.get_tracer("simple-tool-agent")


# ---------------------------------------------------------
# 2. Define the tool and its simulated database
# ---------------------------------------------------------

ORDERS = {"1001": {"status": "shipped",
                   "estimated_delivery": "tomorrow"},
          "1002": {"status": "processing",
                   "estimated_delivery": "not yet available"}}


def get_order_status(order_id: str) -> dict:
    """Look up an order in the fictional order database."""
    with tracer.start_as_current_span("get_order_status") as span:
        span.set_attribute("tool.name", "get_order_status")
        span.set_attribute("tool.order_id", order_id)

        result = ORDERS.get(order_id, {"error": "Order not found"})
        span.set_attribute("tool.success", "error" not in result)
        return result


TOOLS = [{"type": "function",
          "function": {"name": "get_order_status",
                       "description": ("Look up an order's status and estimated delivery. "
                                       "Use this when the user asks about a specific order."),
                       "parameters": {"type": "object",
                                      "properties": {"order_id": {"type": "string",
                                                                  "description": "The order number."}},
                                      "required": ["order_id"],
                                      "additionalProperties": False},
                       "strict": True}
        }]


# ---------------------------------------------------------
# 3. Implement the tool-using agent
# ---------------------------------------------------------

def run_agent(question: str) -> dict:
    """Answer a customer question, using the order tool when appropriate."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a customer support assistant. For questions about a "
                "specific order's status or delivery, call get_order_status "
                "before answering. For general questions, do not call the tool. "
                "If you are unable to answer, state that you can help check an "
                "order's status given its order number. Never invent order "
                "information. Answer clearly."
            ),
        },
        {"role": "user", "content": question},
    ]

    tool_called = False
    tool_names = []

    for _ in range(4):
        response = openai_client.chat.completions.create(model="qwen2.5:14b",
                                                         messages=messages,
                                                         tools=TOOLS,
                                                         tool_choice="auto",
                                                         temperature=0)
        message = response.choices[0].message

        if not message.tool_calls:
            return {"answer": message.content or "",
                    "tool_called": tool_called,
                    "tool_names": tool_names}

        # Preserve the assistant's tool-call message in the conversation.
        messages.append(message.model_dump(exclude_none=True))

        for tool_call in message.tool_calls:
            if tool_call.function.name != "get_order_status":
                raise ValueError(f"Unexpected tool: {tool_call.function.name}")

            args = json.loads(tool_call.function.arguments)
            result = get_order_status(**args)
            tool_called = True
            tool_names.append(tool_call.function.name)

            messages.append({"role": "tool",
                             "tool_call_id": tool_call.id,
                             "content": json.dumps(result)})

    raise RuntimeError("Agent exceeded the maximum number of steps")
