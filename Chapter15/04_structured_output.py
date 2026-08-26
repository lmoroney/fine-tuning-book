"""
04_structured_output.py
-----------------------
Force your model to return valid JSON that matches a schema, every time.

An expert model is only useful in a pipeline if the software downstream can
parse its output. "Please respond in JSON" prompting fails often enough to
break production. Guided decoding solves it at the sampling level: the server
constrains the model so the ONLY tokens it can emit are ones that keep the
output valid against your schema. The result parses 100% of the time.

vLLM supports this through the OpenAI `response_format` / `guided_json` field.
Ollama supports it through its `format` field. This example uses vLLM.

    bash 02_vllm_serve.sh              # in another terminal
    python 04_structured_output.py
"""
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# The contract the rest of your system depends on.
TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "condition": {"type": "string"},
        "urgency": {"type": "string", "enum": ["routine", "urgent", "emergency"]},
        "refer_to_specialist": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": ["condition", "urgency", "refer_to_specialist", "reasoning"],
}

resp = client.chat.completions.create(
    model="medical-gemma",
    messages=[{
        "role": "user",
        "content": "58yo male, crushing chest pain radiating to left arm, "
                   "diaphoretic, 20 minutes. Triage this.",
    }],
    temperature=0.2,
    # vLLM extension: constrain generation to this JSON schema.
    extra_body={"guided_json": TRIAGE_SCHEMA},
)

raw = resp.choices[0].message.content
data = json.loads(raw)        # guaranteed to parse; that is the whole point
print(json.dumps(data, indent=2))

assert data["urgency"] in {"routine", "urgent", "emergency"}
print("\nParsed and schema-valid. Safe to hand to the next stage.")
