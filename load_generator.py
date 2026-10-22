from phoenix.otel import register
from phoenix.evals.llm import LLM
from openai import OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor
import time

PROJECT_NAME = "english-quality-tests"

# Configure Phoenix project and tracing
tracer_provider = register(project_name=PROJECT_NAME, endpoint="http://localhost:6006/v1/traces", protocol="http/protobuf")

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

llm_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

system_prompt = """You are a specialist in European culture, history, traditions, languages, art, 
                   architecture, cuisine, and social customs
                   Answer the user's question directly in clear, natural English. 
                   Include relevant context when useful, but avoid unnecessary details.
                   Keep every answer within 200 words."""

while True:
    question = "Tell me anything about an European city of your choice"

    try:
        response = llm_client.chat.completions.create(
            model="qwen2.5:14b",
            messages=[{"role": "system",
                       "content": system_prompt,},                
                      {"role": "user",
                       "content": question}])
        answer = response.choices[0].message.content
        print(f"Answer: {answer}")
        tracer_provider.force_flush()
    except Exception as exc:
        print(f"Failed: {type(exc).__name__}: {exc}")
    time.sleep(20)
