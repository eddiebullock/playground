# Paper 3 draft: frontier models as social intermediaries

Working title: **"When the assistant writes the reply: a pre-registered
red-team of frontier language models on crisis, fraud, and consent"**

Target: *Nature Machine Intelligence* or *Nature Human Behaviour* (first
choice if human double-coding and six model snapshots are complete);
FAccT / CHI / AIES as the strong fallback. Status: protocol, 2026-10-29.
This paper does **not** evaluate Gloss.

## Question

Autistic adults already use consumer chatbots to interpret messages and
draft replies. Wohn et al. (CHI 2026) showed that disclosing autism
shifts those replies toward avoidance. The open safety question is
coarser and more serious: in that same role, how often do current
frontier models produce an output that would *do the harm* — skip a
crisis referral, comply with a scam, or talk someone past a no?

## Contribution (if the confirmatory run holds)

1. A behavioural endpoint (harmful-action) rather than a keyword hit rate,
   coded by two humans, on a sealed item set.
2. Evidence that an autistic-user system prompt is not a safety
   intervention and may increase scam and boundary harm (over-helpfulness).
3. A displacement test: extra safety text can move errors onto hard
   negatives or onto another harm type rather than reducing total harm.
4. An independent academic eval of deployed assistants, not a vendor
   card and not a prototype.

## Design in one paragraph

180 held-out items (distress, scam, boundary; 40 true-positive and 20
hard-negative each). Six model snapshots. Two primary personas (generic
assistant vs social assistant for an autistic adult) plus an exploratory
safety-augmented persona. Temperature 0. Mixed-effects logit on
harmful-action with item random intercepts. Full protocol:
`docs/research/05_frontier_safety.md`.

## Results

None yet. Development items exist for codebook freeze and harness
dry-run only. Do not quote development rates in the paper.

## Needed for submission

- [EB] OSF snapshot of `experiments/e11_frontier_safety/prereg.md`.
- [EB] API keys / spend for OpenAI, Anthropic, Google, xAI. Open-weight
  70B if available; otherwise drop that arm.
- Two human raters, kappa on the 60-output calibration set, then holdout.
- Advisory read of the item *categories* (not the scam full texts) by at
  least one autistic adult. Eddie arranges this.
- Dual-use pass: paper prints categories, not a scam pack.
