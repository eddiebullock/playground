"""A two-call smoke plan: one multiple-choice item and one seeded generation.

The plan is what a GPU job should run. It does not download a model.
"""

from __future__ import annotations

from scog.infer.vllm_server import generation_request, logprob_request, multiple_choice_prompt


def smoke_calls(model: str) -> list[dict]:
    if not model:
        raise ValueError("smoke test model name is empty")
    prompt = multiple_choice_prompt(
        "Which word is closest to 'happy'?",
        ["sad", "joyful", "angry", "tired"],
    )
    mcq = logprob_request(prompt, ["A", "B", "C", "D"])
    mcq.update(
        {
            "model": model,
            "item": "smoke-mcq",
            "condition": "smoke",
            "permutation": 0,
            "seed": 0,
        }
    )
    gen = generation_request("Say hello in one word.", seed=20261029, max_tokens=8)
    gen.update(
        {
            "model": model,
            "item": "smoke-gen",
            "condition": "smoke",
            "permutation": 0,
            "seed": 20261029,
        }
    )
    return [mcq, gen]
