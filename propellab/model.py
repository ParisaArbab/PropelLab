"""Structured inference with a local server or a configured Responses API."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .evidence import COMPONENTS


def object_schema(properties: dict) -> dict:
    return {
        "type": "object", "properties": properties, "required": list(properties),
        "additionalProperties": False,
    }


DIAGNOSIS_SCHEMA = object_schema({
    "ranking": {"type": "array", "items": {"type": "string", "enum": list(COMPONENTS)}, "maxItems": 3},
    "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    "reason": {"type": "string"},
})

PLAN_SCHEMA = object_schema({
    "tests": {"type": "array", "items": object_schema({
        "requirement": {"type": "string"},
        "cycle": {"type": "string", "enum": ["udds", "hwfet", "us06"]},
        "rationale": {"type": "string"},
    })},
})


@dataclass(frozen=True)
class Response:
    data: dict
    input_tokens: int
    output_tokens: int
    seconds: float
    prompt_sha256: str


class LanguageModel:
    def __init__(self, url: str, model: str, api: str, timeout: float):
        self.url = url.rstrip("/")
        self.model = model
        self.api = api
        self.timeout = timeout

    def complete(self, system: str, user: dict, schema: dict, output_limit: int) -> Response:
        payload = json.dumps(user, separators=(",", ":"), allow_nan=False)
        if self.api == "responses":
            body = {
                "model": self.model, "instructions": system, "input": payload,
                "store": False, "max_output_tokens": output_limit,
                "text": {"format": {"type": "json_schema", "name": "propellab",
                                    "strict": True, "schema": schema}},
            }
            endpoint = "responses"
        else:
            body = {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": payload}],
                "temperature": 0.0, "max_tokens": output_limit,
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "propellab", "strict": True, "schema": schema,
                }},
            }
            endpoint = "chat/completions"
        headers = {"Content-Type": "application/json"}
        key = os.environ.get("PROPELLAB_MODEL_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = Request(f"{self.url}/{endpoint}", data=json.dumps(body).encode(), headers=headers)
        start = time.monotonic()
        for attempt in range(3):
            try:
                with urlopen(request, timeout=self.timeout) as connection:
                    reply = json.load(connection)
                break
            except HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise RuntimeError(f"Model endpoint returned HTTP {exc.code}") from exc
            except (URLError, TimeoutError) as exc:
                if attempt == 2:
                    raise RuntimeError(f"Model endpoint unavailable: {exc}") from exc
            time.sleep(1 + attempt)
        elapsed = time.monotonic() - start
        usage = reply["usage"]
        if self.api == "responses":
            if reply.get("status") != "completed":
                raise ValueError(f"Incomplete model response: {reply.get('status')}")
            content = "".join(
                part["text"] for output in reply["output"] if output["type"] == "message"
                for part in output["content"] if part["type"] == "output_text"
            )
            input_tokens, output_tokens = usage["input_tokens"], usage["output_tokens"]
        else:
            choice = reply["choices"][0]
            if choice["finish_reason"] != "stop":
                raise ValueError(f"Incomplete model response: {choice['finish_reason']}")
            content = choice["message"]["content"]
            input_tokens, output_tokens = usage["prompt_tokens"], usage["completion_tokens"]
        return Response(
            json.loads(content), input_tokens, output_tokens, elapsed,
            hashlib.sha256(f"{system}\n{payload}".encode()).hexdigest(),
        )


DIAGNOSIS_INSTRUCTIONS = """You are triaging one failed toy propulsion controller test.
Identify the primary software component that best explains the observed changes.
The candidate components are:
""" + "\n".join(f"- {name}: {description}" for name, description in COMPONENTS.items()) + """
Each row gives observed minus healthy-reference telemetry for the same second,
stimulus and powertrain. req, energy, cap, cmd and actual are signed hundredths.
temp_flag, diag and gen are integer changes in status codes. Rows can be sparse.
Follow the signal path from an early changed signal to its downstream symptoms.
A temperature dropout or overheat can be intentional and is present in both runs.
The generator exists only on REEV. Rank at most three distinct component IDs.
Cite up to three provided evidence IDs. Keep the reason to one short sentence."""


def diagnose(model: LanguageModel, evidence: dict) -> dict:
    response = model.complete(DIAGNOSIS_INSTRUCTIONS, evidence, DIAGNOSIS_SCHEMA, 230)
    valid_ids = {row.split(",", 1)[0] for row in evidence["paired_trace"]}
    ranking = response.data["ranking"]
    citations = response.data["evidence"]
    if (len(ranking) > 3 or len(set(ranking)) != len(ranking)
            or any(component not in COMPONENTS for component in ranking)
            or (evidence["powertrain"] != "REEV" and "generator_supervisor" in ranking)
            or len(citations) > 3 or any(citation not in valid_ids for citation in citations)
            or (ranking and not citations)):
        raise ValueError("Model returned an unsupported component or evidence reference")
    return {
        **response.data, "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens, "seconds": round(response.seconds, 4),
        "prompt_sha256": response.prompt_sha256,
    }


def draft_plan(model: LanguageModel, requirements: list[dict]) -> tuple[list[dict], dict]:
    instructions = (
        "Draft one test specification per supplied requirement. Choose an EPA drive cycle "
        "from the provided IDs and give a short reason for the choice. Cover every requirement "
        "exactly once. The requested stimulus and pass criteria come from the requirement."
    )
    response = model.complete(instructions, {"requirements": requirements}, PLAN_SCHEMA, 620)
    tests = response.data["tests"]
    required_ids = sorted(requirement["id"] for requirement in requirements)
    if (sorted(test["requirement"] for test in tests) != required_ids
            or any(test["cycle"] not in {"udds", "hwfet", "us06"}
                   or not test["rationale"].strip() for test in tests)):
        raise ValueError("Draft plan failed requirement or cycle validation")
    metadata = {
        "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
        "seconds": round(response.seconds, 4), "prompt_sha256": response.prompt_sha256,
    }
    return tests, metadata
