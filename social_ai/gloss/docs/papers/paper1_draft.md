# Paper 1 draft: literal-biased, not literal-minded

Working title: **"Literal-biased, not literal-minded: separating pragmatic
sensitivity from response bias in language models, and why benchmark
accuracy overstates tools"**

Target: TACL (first choice), ACL/EMNLP main as a fallback. Upgrade path
with human minimal-pair data: *Nature Human Behaviour* / *PNAS* /
*Cognitive Science* (see `docs/PUBLICATION_STRATEGY.md`).
Status: skeleton with the numbers available on 2026-10-25. Slots marked
[ON] are filled by the overnight queue; [EB] need Eddie (human data,
authorship, ethics). Supersedes `docs/paper_e1_outline.md`.

## Abstract (draft)

Language models are often said to read non-literal language "literally",
a claim built on benchmarks in which every correct answer is non-literal.
On such benchmarks a model that is simply biased towards literal readings is
indistinguishable from one that cannot detect non-literal intent. We
introduce minimal-pair sincere controls for the Hu et al. (2023)
pragmatics items: the same remark, with the situation rewritten so that it
is meant literally, for irony, indirect requests and deceits (64 pairs).
Signal-detection analysis across 5 open models from 3 families shows
that the literal tendency is a criterion effect specific to irony: qwen3:8b
distinguishes ironic from sincere versions almost perfectly (paired AUC
0.96), yet chooses the literal reading for half of the ironic ones. A
one-line instruction that people often mean something other than their
words moves the irony criterion from c = 0.96 to 0.18 with no change in
sensitivity (d' 2.21 vs 2.25), raising irony accuracy from 0.51 to 0.85. On
indirect requests and deceits, where the model's criterion was already
neutral, the same instruction creates a non-literal bias: truthful
statements are read as deceptive on 53% of sincere deceit pairs. The
same lens explains a second artefact. When the question names the construct
("is this sarcastic?"), models answer yes to almost every line in MUStARD.
After temperature scaling, abstaining below a calibrated 0.7 gives 0.91
accuracy on 71% of items, above the human mean. Finally, when the same model
writes its own readings, as an assistant would, the intended meaning comes
first in only 0.50-0.56 of cases. Writing the readings, not ranking them, is
the bottleneck, and LLM-as-judge grading of such outputs agrees with blind
labels on only 0.74 of readings. [ON: external replication on author-labelled
real tweets, iSarcasmEval.]

## 1. Introduction (points to make)
1. Claim in the literature: LLMs are literal, or struggle with irony and
   implicature (Hu et al. 2023; Ruis et al. 2023; Sravanthi et al. 2024).
   Benchmarks are built of non-literal items.
2. The confound is old news in psychophysics and has a fix: signal
   detection separates sensitivity and bias. Applied here, it needs items
   where the literal answer is right, matched to items where it is wrong.
3. Why it matters beyond NLP: tools that read messages for people (e.g.
   autistic adults, who already use LLMs to check tone) fail differently
   depending on which it is. A bias can be moved with an instruction; a
   sensitivity gap cannot. Over-reading sincere messages is a harm in its own
   right (it can reinforce anxious readings).
4. Contributions: (i) minimal-pair sincere controls for three phenomena
   ([ON: 64] pairs), released; (ii) d'/c analysis across models and prompts;
   (iii) demand-characteristic effect on sarcasm detection; (iv) calibrated
   selective accuracy against human data; (v) the benchmark-to-tool gap,
   with a judge-reliability check.

## 2. Materials
- Hu et al. (2023) items: 7 phenomena, 169 items, human accuracy from 374
  participants. 5 option orders each.
- Minimal pairs: Irony 25, IndirectSpeech 20, Deceits 19 (one Deceits item
  cannot be made sincere without changing the remark; reported). Written by
  an LLM assistant under fixed rules (remark verbatim, question unchanged,
  only facts change, no loud cues), verified by script. [EB: validation by 2
  human raters blind to version; target agreement with intended answer
  >= 0.9; drop failing pairs before analysis.] [EB: optional human hit/FA
  rates on the pairs.]
- MUStARD: balanced 200-clip subset (seed 42), text only, 2 orders.
- iSarcasmEval (Abu Farha et al. 2022): 80 sarcastic tweets with the
  author's own statement of what they meant, 80 sincere tweets; frozen
  before any run (sha256 in `experiments/e8_isarcasm/items.sha256`).

## 3. Methods
- Option probabilities from first-token log-probs after an "Answer:"
  prefill, averaged over orders (removes position bias). Local inference
  (Ollama, temperature 0).
- Models: qwen3 4B, 8B; llama3.1 8B; gemma3 4B, 12B.
- Prompts: direct ("Answer with the number..."); stance (one sentence:
  people often mean something other than their words, and often exactly
  what they say). The stance sentence was written after the E1 direct
  results and before any minimal-pair run.
- SDT on pairs: q = p(non-literal)/(p(non-literal) + p(literal)); H, F at
  q > 0.5 with log-linear correction; d', c; paired AUC; bootstrap CIs over
  pairs (`experiments/e1_pragmatics/sdt.py`).
- Calibration: temperature scaling with 5-fold CV over items; ECE;
  selective accuracy vs coverage.
- Product study (E4/E8): the assistant writes 2-4 readings (literal first),
  then the same scorer weighs them. Each reading was labelled blind (runs
  mixed, order shuffled, no rank shown) for whether it states the intended
  meaning. [EB: second human rater on a 20% sample, kappa.]

## 4. Results (numbers so far)

### 4.1 Accuracy on Hu items (order-averaged per trial, 845 trials)
| Model | Direct | Stance | Human |
|---|---|---|---|
| qwen3:4b | 0.548 | 0.646 | 0.863 |
| qwen3:8b | 0.683 | 0.780 | 0.863 |
| llama3.1:8b | 0.676 | 0.703 | 0.863 |
| gemma3:4b | 0.586 | 0.611 | 0.863 |
| gemma3:12b | 0.814 | 0.844 | 0.863 |

qwen3:8b by phenomenon, direct -> stance: Irony 0.51 -> 0.85; Indirect
0.81 -> 0.91; Deceits 0.76 -> 0.87; Maxims 0.59 -> 0.74; Metaphor 0.80 ->
0.79; Humour 0.56 -> 0.61; Coherence 0.75 -> 0.75.

### 4.2 Sensitivity and bias (irony pairs, n = 25)
| Model / prompt | acc ironic | acc sincere | d' [95% CI] | c [95% CI] | paired AUC |
|---|---|---|---|---|---|
| qwen3:4b direct | 0.48 | 0.96 | 1.53 [0.96, 2.31] | 0.81 [0.43, 1.26] | 0.96 |
| qwen3:4b stance | 0.64 | 0.96 | 1.92 [1.33, 2.74] | 0.61 [0.23, 1.06] | 1.00 |
| qwen3:8b direct | 0.56 | 0.96 | 2.21 [1.73, 2.74] | 0.96 [0.70, 1.21] | 0.96 |
| qwen3:8b stance | 0.84 | 0.92 | 2.25 [1.62, 3.37] | 0.18 [-0.25, 0.70] | 1.00 |
All 64 pairs by phenomenon, qwen3:8b (2026-10-25 overnight):
| Prompt | phenomenon (pairs) | acc non-literal | acc sincere | d' | c |
|---|---|---|---|---|---|
| direct | Irony (25) | 0.56 | 0.96 | 2.21 | +0.96 |
| direct | Indirect requests (20) | 0.85 | 0.75 | 1.60 | -0.17 |
| direct | Deceits (19) | 0.74 | 0.74 | 1.20 | 0.00 |
| stance | Irony | 0.84 | 0.92 | 2.25 | +0.18 |
| stance | Indirect requests | 0.90 | 0.65 | 1.55 | -0.41 |
| stance | Deceits | 0.90 | 0.47 | 1.09 | -0.61 [-1.21, -0.14] |
| direct / stance | All 64 | 0.70 / 0.88 | 0.83 / 0.70 | 1.51 / 1.65 | +0.23 / -0.30 |
qwen3:4b, all 64: d' 1.18 -> 1.47, c +0.40 -> +0.25.

Across all five models (direct -> stance):
| Model | Irony c | Irony d' | Deceits acc sincere (direct) | All-64 balanced acc |
|---|---|---|---|---|
| qwen3:4b | +0.81 -> +0.61 | 1.53 -> 1.92 | 0.74 | 0.68 -> 0.76 |
| qwen3:8b | +0.96 -> +0.18 | 2.21 -> 2.25 | 0.74 | 0.77 -> 0.79 |
| llama3.1:8b | +0.81 -> -0.25 | 2.52 -> 2.11 | 0.26 | 0.75 -> 0.76 |
| gemma3:4b | +0.76 -> +0.45 | 1.62 -> 2.25 | 0.42 | 0.68 -> 0.74 |
| gemma3:12b | +0.37 -> -0.39 | 1.86 -> 2.38 | 0.47 | 0.79 -> 0.81 |
The direct irony criterion is literal-leaning in every model; the deceit
criterion is family-specific (Qwen neutral; Llama and Gemma read truthful
statements as deceptive by default).

Reading: the prompt barely changes sensitivity anywhere. The literal bias is
specific to irony; for indirect requests and deceits the direct criterion is
already near neutral, and the global instruction pushes it past neutral, so
that truthful statements are read as deceptive or excusing on half the
sincere deceit pairs. Criteria are phenomenon-specific; a prompt moves them
all together. This is the paper's main claim.

### 4.3 Naming the construct
MUStARD, "is this line sarcastic or sincere?": accuracy 0.51-0.54 for every
Qwen model and prompt; "sarcastic" recall 0.95-0.98, precision 0.51-0.52. The
models' c is strongly liberal here, the opposite sign to 4.2 on the same
kind of content. Interpretation: naming sarcasm in the question licenses it.
Replicates in every family: "sarcastic" recall 0.97-0.99 for llama3.1:8b,
gemma3:4b and gemma3:12b.

### 4.4 Calibration and abstention (qwen3:8b stance)
ECE 0.19 raw -> 0.044 after temperature scaling (T = 3.4). At calibrated
>= 0.7: coverage 71%, accuracy 0.91 (human mean 0.86 over all items).
Item entropy vs human disagreement rho 0.51 (direct 0.37).

### 4.5 From benchmark to tool
| Measure (103 Hu items, blind labels) | Shipped engine | Prompt v2 (held-out half, 52) |
|---|---|---|
| Intended meaning ranked first | 0.495-0.534 | 0.558 |
| Intended meaning written at all | 0.68-0.74 | 0.77 |
| Ranked first when written | 0.75 | |
| Sincere controls: plain reading first | 0.72 | 0.75 |
LLM judge vs blind labels: agreement 0.74, precision 0.62.
[ON: iSarcasmEval replication; write-more-show-fewer.]

## 5. Discussion points
- "Literal-minded" is the wrong diagnosis for these models: they detect the
  non-literal reading and decline to choose it. Prompt fixes work because
  they move the criterion, and they move it for sincere items too, so they
  trade false alarms for hits. Any evaluation of pragmatic prompting must
  report both.
- The literalism-autism parallel drawn in earlier work deserves the same
  scrutiny. Whether autistic readers' literal answers on such tasks reflect
  detection or criterion is an open, testable question (double empathy;
  Milton 2012), and E7 tests it directly. We make no claim about it here. [EB:
  framing check with the advisory group.]
- Practical: report d'/c with minimal pairs; never name the construct in
  the question; grade generative outputs with blind human labels, not a
  judge.

## 6. Limitations
English vignettes; minimal pairs written by an LLM (validation pending);
multiple-choice option sets fixed by the original authors; small open
models only (frontier models excluded by cost); one rater for product
labels so far.

## 7. Release
Minimal pairs, iSarcasm item ids (not tweet text), code, all per-item
outputs. [EB: OSF pre-registration of the human validation study.]
