"""E1: pragmatic interpretation and calibrated deferral in small on-device LLMs.

Question: can a free, local model read non-literal language (irony, indirect
speech, deceit, maxims, metaphor, humour, coherence; sarcasm in dialogue) well
enough, and does its confidence track both its own errors and human
disagreement, so that a tool can say "possibly" and defer when it should?

Reads the option distribution from first-token log-probs (one call per item
per option-order seed), via the Ollama chat API. Stdlib only.

Usage:
  python run.py --model qwen3:4b --task hu
  python run.py --model qwen3:4b --task mustard
"""

import argparse
import ast
import csv
import json
import math
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HU = ROOT / "data/raw/lm-pragmatics"
MUSTARD = ROOT / "data/raw/MUStARD/data/sarcasm_data.json"
OUT = Path(__file__).resolve().parent / "results"

SYSTEM = "Answer with the number of the best option only. No other text."
HU_TASKS = ["Irony", "IndirectSpeech", "Deceits", "Maxims", "Metaphor", "Humour", "CoherenceInference"]


SYSTEM_REASON = "Think briefly about what the speaker really means, then give the number of the best option."

# Raw templates. The answer is read after an "Answer: " prefill, so the next
# token is the option number. In direct mode Qwen3 gets an empty think block;
# in reason mode the model writes its reasoning first (Qwen3 inside <think>,
# Gemma as plain text), then the same prefill is appended.
TURN = {
    "qwen": "<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n",
    "gemma": "<start_of_turn>user\n{s}\n\n{u}<end_of_turn>\n<start_of_turn>model\n",
}


def family(model):
    return "gemma" if model.startswith("gemma") else "qwen"


def generate(model, prompt, num_predict, logprobs=False, stop=None, timeout=600):
    body = {
        "model": model,
        "prompt": prompt,
        "raw": True,
        "stream": False,
        "options": {"temperature": 0, "num_predict": num_predict, "seed": 0},
    }
    if logprobs:
        body.update({"logprobs": True, "top_logprobs": 20})
    if stop:
        body["options"]["stop"] = stop
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    resp["message"] = {"content": resp.get("response", "")}
    return resp


def chat(model, prompt, mode="direct"):
    fam = family(model)
    if mode == "direct":
        head = TURN[fam].format(s=SYSTEM, u=prompt + "\n\nReply with the option number only.")
        if fam == "qwen":
            head += "<think>\n\n</think>\n\n"
        return generate(model, head + "Answer: ", 2, logprobs=True)
    head = TURN[fam].format(s=SYSTEM_REASON, u=prompt)
    if fam == "qwen":
        head += "<think>\n"
        thought = generate(model, head, 400, stop=["</think>"])
        # Ollama splits Qwen3 reasoning into its own field.
        text = (thought["response"] or thought.get("thinking", "")).strip()
        resp = generate(model, head + text + "\n</think>\n\nAnswer: ", 2, logprobs=True)
    else:
        thought = generate(model, head, 300, stop=["Answer:"])
        text = thought["response"].strip()
        resp = generate(model, head + text + "\n\nAnswer: ", 2, logprobs=True)
    resp["reasoning"] = text
    resp["total_duration"] = resp.get("total_duration", 0) + thought.get("total_duration", 0)
    resp["reason_tokens"] = thought.get("eval_count", 0)
    return resp


def option_dist(resp, n_opts):
    """Probability over options 1..n from the first answer token; None if unreadable."""
    lps = resp.get("logprobs") or []
    for tok in lps:
        if tok["token"].strip() == "":
            continue
        mass = {}
        for alt in tok.get("top_logprobs", []):
            t = alt["token"].strip().rstrip(")").rstrip(".")
            if t in {str(k) for k in range(1, n_opts + 1)}:
                k = int(t)
                mass[k] = mass.get(k, 0.0) + math.exp(alt["logprob"])
        if mass:
            z = sum(mass.values())
            return [mass.get(k, 0.0) / z for k in range(1, n_opts + 1)]
        break
    text = resp["message"]["content"].strip()
    if text[:1] in {str(k) for k in range(1, n_opts + 1)}:
        p = [0.0] * n_opts
        p[int(text[0]) - 1] = 1.0
        return p
    return None


def hu_items(task, seed):
    path = HU / f"prompts/{task}_prompts_seed{seed}_examples0.csv"
    for row in csv.DictReader(open(path)):
        order = ast.literal_eval(row["randomized_option_order"])
        yield {
            "id": f"{task}:{row['item_id']}",
            "task": task,
            "seed": seed,
            "prompt": row["prompt"].replace("Answer: ", "").strip(),
            "n_opts": len(order),
            "true": int(row["randomized_true_answer"]),
            # position i (1-based) shows original option order[i-1]
            "order": order,
            "labels": ast.literal_eval(row["randomized_labels_complex"]),
        }


def mustard_items(seed):
    data = json.load(open(MUSTARD))
    for key, d in data.items():
        ctx = "\n".join(f"{s.title()}: {u}" for s, u in zip(d["context_speakers"], d["context"]))
        opts = ["Sarcastic: they mean something different from, often the opposite of, what they literally say.",
                "Sincere: they mean what they literally say."]
        flip = seed % 2 == 1
        if flip:
            opts = opts[::-1]
        prompt = (
            "Here is a short exchange from a conversation.\n\n"
            f"{ctx}\n{d['speaker'].title()}: {d['utterance']}\n\n"
            f"Is {d['speaker'].title()}'s last line sarcastic or sincere?\n"
            f"1) {opts[0]}\n2) {opts[1]}"
        )
        true = 1 if d["sarcasm"] else 2
        if flip:
            true = 3 - true
        yield {"id": f"mustard:{key}", "task": "MUStARD", "seed": seed, "prompt": prompt,
               "n_opts": 2, "true": true, "sarcasm": d["sarcasm"], "show": d["show"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=["hu", "mustard"], required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--mode", choices=["direct", "reason"], default="direct")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{args.task}_{args.model.replace(':', '-').replace('/', '-')}{'' if args.mode == 'direct' else '_reason'}.jsonl"
    done = set()
    if out.exists():
        done = {(r["id"], r["seed"]) for r in map(json.loads, open(out))}

    if args.task == "hu":
        items = [it for s in range(args.seeds) for t in HU_TASKS for it in hu_items(t, s)]
    else:
        items = [it for s in range(min(args.seeds, 2)) for it in mustard_items(s)]

    t0 = time.time()
    with open(out, "a") as f:
        for i, it in enumerate(items):
            if (it["id"], it["seed"]) in done:
                continue
            resp = chat(args.model, it["prompt"], args.mode)
            p = option_dist(resp, it["n_opts"])
            rec = {k: v for k, v in it.items() if k != "prompt"}
            rec.update({"p": p, "raw": resp["message"]["content"][:20],
                        "ms": resp.get("total_duration", 0) / 1e6, "mode": args.mode})
            if "reasoning" in resp:
                rec.update({"reasoning": resp["reasoning"], "reason_tokens": resp["reason_tokens"]})
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % (50 if args.mode == "direct" else 10) == 0:
                print(f"{i}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    print("done", out)


if __name__ == "__main__":
    main()
