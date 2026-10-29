# Publication strategy

Written 2026-10-25 (evening) by Claude for Eddie. Question asked: can the
Gloss results be organised into publishable studies, and how do we make
them count in top-tier venues? Short answer: yes, three papers are
realistic from work that runs on this laptop, and two of them can reach a
top venue *if* they get human data that only Eddie can authorise. The
rest of this file says which papers, what each claims, what is missing,
and what the overnight runs (`docs/OVERNIGHT_PLAN.md`) add.

## 1. What reviewers at top venues will ask, and where we stand

| Reviewer question | Now | Fix |
|---|---|---|
| Is this one model's quirk? | Only Qwen3 4B and 8B | Add 2-3 families (Llama, Gemma; free, local). Overnight. |
| Who wrote the labels? | Claude (an LLM) wrote all "hand" labels and the 25 sincere pairs | Human raters: Eddie plus one more, blind, on a sample (kappa). Then crowd or volunteer raters for the minimal pairs. **Needs Eddie.** |
| Were analyses fixed in advance? | Hypotheses are dated in DECISIONS, not registered | OSF pre-registration for every confirmatory run from now on. **Eddie's account.** |
| Is the test set contaminated / tuned on? | E4 both halves seen | New held-out data from outside (iSarcasmEval, author-labelled real tweets). Overnight. |
| Real messages, or puzzle stories? | Hu et al. stories only | iSarcasmEval (real tweets, the author's own intended meaning). Overnight. |
| Effect sizes and uncertainty? | CIs everywhere, small n in places | Larger audit sets; report d', criterion, paired tests. Overnight. |
| Community involvement? | Focus groups done; no autistic co-author yet | Advisory group and a co-author or named contributors. Required by *Autism*, *Autism in Adulthood*. **Needs Eddie.** |

## 2. The papers

### Paper 1. "Pragmatic competence or a taste for the non-literal? Sensitivity, bias and calibration in open language models" (E1, E1b, E3, sincere minimal pairs, MUStARD, E4 as the product test)

**The contribution the field does not have.** Every item in the standard
pragmatics benchmark (Hu et al. 2023) has a non-literal correct answer. A
model that simply prefers non-literal readings scores well. Our minimal
pairs (same remark, context rewritten so it is sincere) separate the two
for the first time, in signal-detection terms: sensitivity (d') versus
bias (criterion c). The data we already have show why it matters:

| qwen3:8b, irony vs sincere pairs (25) | hit (irony) | correct rejection (sincere) | d' | c |
|---|---|---|---|---|
| Direct | 0.51 | 0.976 | about 2.0 | strongly literal |
| Pragmatic-stance prompt | 0.85 | 0.904 | about 2.3 | near neutral |

So the famous "LLMs read irony literally" result is mostly a *criterion*
effect, and the one-line prompt that "fixes" it mostly moves the criterion,
with a smaller real gain in sensitivity. (These figures are approximate and
are recomputed per item in the paper.) Three further findings support it:
- **Naming the construct creates a yes-bias.** Asked "is this line
  sarcastic?", both models say yes to 96-98% of MUStARD lines, sarcastic or
  not (accuracy at chance). A demand-characteristic effect, with a clear
  lesson for how sarcasm benchmarks are posed.
- **Calibrated abstention reaches human-level accuracy on what it
  answers.** 8B with stance and temperature scaling: 0.91 accuracy on 71%
  of items (humans 0.86 on all).
- **Benchmarks overstate tools.** When the same model writes its own
  readings (the product, E4), the intended meaning comes top only about
  half the time (0.50-0.56), against 0.78 in multiple choice. Generation
  misses the right reading on a quarter of items. LLM-as-judge grading
  agreed with blind labels only 0.74, so judge-only product evaluations in
  the literature are suspect.

**What it needs to reach a top venue.**
1. Minimal pairs across phenomena, not just irony: indirect requests,
   deceits (true statements), maxims (answers that do answer). Claude
   writes them overnight; humans must validate them.
2. 4-5 models across 3 families. Overnight (free local models).
3. **Human data on the minimal pairs** (the step that changes the tier).
   With human hit and correct-rejection rates on the same pairs, the paper
   can say whether models are *more* or *less* literal than people in
   criterion, not just in accuracy. About 100 raters x 40 items online.
   Prolific costs about GBP 300-500: Eddie's decision under the no-spend
   rule. A free route: ARC participant panel or student volunteers.
4. External real-message test: iSarcasmEval (author-labelled sarcasm with
   the author's own rephrasing of what they meant). Overnight.

**Venues.** With 1, 2, 4: TACL or ACL/EMNLP main (strong, realistic).
With 3 as well: *Nature Human Behaviour* or *PNAS* become arguable if
framed around human vs machine pragmatic bias; *Cognitive Science* or
*Open Mind* as the safe cognitive-science route. Recommended: write it
for TACL now, keep the human study as the upgrade path.

### Paper 2. "Telling an AI you are autistic: disclosure effects on social advice, and why prompt fixes move them rather than remove them" (E4 audits)

**Contribution.** Wohn et al. (CHI 2026) showed advice drifting towards
avoidance after autism disclosure in frontier chat models. Our audit
asks the same of the readings *and* the advice in a tool built not to
drift, with a noise floor from repeated seeds, and finds: no drift for
autism disclosure; a small, real drift for anxiety disclosure (+5 points
apologetic replies, CI excludes 0); and that the first prompt fix
**relabelled** the drift into avoidance (+8 points hold-back, CI excludes 0)
rather than removing it. "Whack-a-mole debiasing" is a general and
under-reported failure of prompt-level fairness fixes. The fix-2 result
(running) decides whether a wording fix can work at all or only a
structural one (keep the disclosure out of the call).

**What it needs.**
1. More scenarios (12 to 48) and more disclosures: autistic, ADHD,
   anxious, depressed, and a **placebo disclosure** ("I am left-handed")
   that should do nothing. The placebo is what makes a reviewer believe the
   effect is about the condition, not about any extra sentence. Overnight
   for the readings audit; the replies audit is slower (hours per seed).
2. 3+ models, so the paper is about LLMs, not one prompt. Overnight.
3. The reply classifier is the 8B model itself. Validate it against blind
   human labels on a sample (kappa), or the paper cannot rest on it.
   Claude can do a first pass; Eddie plus a second rater for the paper.
4. Frontier models would make it far more cited (that is where users are),
   but they cost money. Free tiers excluded by the rules. Eddie's call.

**Venues.** FAccT, CHI, AIES or ASSETS. FAccT fits the displacement
finding best; ASSETS fits the autism framing; CHI if it includes the
design rule and the product context. Short paper version: CHI late-breaking
work this cycle.

### Paper 3. Frontier models as social intermediaries: crisis, fraud, consent (E11)

**This is now the safety paper. It does not evaluate Gloss.** Protocol:
`docs/research/05_frontier_safety.md`. Draft:
`docs/papers/paper3_frontier_safety.md`.

**Contribution.** Independent, pre-registered red-team of consumer
frontier chat models in the role autistic adults already give them
(interpret this message, draft a reply). Primary endpoint is
human-coded *harmful-action* on sealed true-positive items: missed
crisis referral, scam-complying draft, coaching past a no. Second
question: does an autistic-user persona increase those harms. Third:
do extra safety instructions displace rather than reduce total harm.

**Why it can reach a top journal.** Deployed systems, not a prototype;
behavioural endpoint, not keyword hits; named high-stakes population;
pre-registered mixed logit; human kappa. Venue path: *Nature Machine
Intelligence* or *Nature Human Behaviour* if six snapshots and double
coding land; FAccT / CHI / AIES otherwise.

**What it needs.** Eddie: OSF snapshot, API spend, two raters, advisory
read of item categories. Agents: no holdout scoring, no paid calls
without `E11_ALLOW_PAID=1`. Expand confirmatory N from the sealed 36
to 180 without peeking at model outputs.

### Paper 3b. Study 5 protocol, then results (live HUD, later)

Protocol: *JMIR Research Protocols* or *Pilot and Feasibility Studies*
after PREC approval, rewritten around Gloss Live (not paste-a-message).
Results: *Autism in Adulthood* or *Autism*. Do not recruit until Eddie
says so.

### Paper 4 (Eddie's, not Claude's). Focus groups: what autistic adults want from AI social support

Sessions 2-4 already contain a qualitative paper (reflexive thematic
analysis; themes T1-T11 in `docs/research/04_focus_groups.md`). Venue:
*Autism in Adulthood*, CHI or ASSETS. Claude has not quoted participants
and should not draft the analysis; it is Eddie's data and method.

### Paper 5 (highest ceiling, needs ethics). E7: do models read like autistic or non-autistic people?

Autistic and non-autistic raters on the same items (including the minimal
pairs). If autistic raters differ from non-autistic raters in criterion
(more literal, or not), and models align with one group, that is a
double-empathy result about machines, publishable in *Nature Human
Behaviour*-class venues and highly cited. Combine with Paper 1's human
study to share one data collection. Needs PREC and recruitment.

## 3. Recommended order

1. Paper 3 (E11) is the new high-ceiling safety paper. Freeze codebook,
   OSF, then paid model runs when Eddie authorises them.
2. Paper 1 to TACL: overnight runs already supply breadth; Eddie and one
   rater validate the minimal pairs.
3. Paper 2 to FAccT or CHI LBW: classifier validation still needed.
4. Study 5 protocol after PREC, about the live HUD.

## 4. Things that would weaken every paper, to avoid

- Claiming Claude's labels are "human" labels. They are LLM labels. Call
  them "expert-written labels by an LLM assistant, validated on a sample
  by two human raters (kappa = ...)" once the validation exists.
- Tuning prompts on test items. Held-out rules in `PLAN.md` section 7.
- Reporting the judge-only numbers.
- Clinical or deficit framing. Use "communication assistant", "readings",
  double empathy; no "compensate for".
