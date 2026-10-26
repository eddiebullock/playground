// Fixed safety layer, independent of the model. E9 (2026-10-26) showed that
// qwen3:8b ignores the boundary instruction (empty in 40/40 cases), ignores
// distress the user writes about themselves (0/5), reads phishing as genuine
// and offers replies that comply ("I'll confirm my card details now"). A prompt
// line cannot be trusted with these, so they are detected with patterns on the
// input and answered with fixed text. Patterns were written from general
// knowledge before looking at held-out items (experiments/e9b_safety_heldout);
// false positives are measured on the E4, E10 and audit sets.
//
// Wording of the notes is a placeholder until Eddie and the advisory group
// review it [EB]. Numbers are UK (Samaritans, 999).

export type SafetyInput = { mode: "them" | "me"; context?: string; text: string; about?: string; own?: string };

export type Safety = {
  /** The user has written about themselves in a way that suggests distress or risk. */
  distress?: string;
  /** The message looks like a known scam pattern. */
  scam?: string;
  /** The user has said no and the other person keeps pushing. */
  boundary?: string;
};

export const DISTRESS_NOTE =
  "It sounds like you are having a really hard time right now. You do not have to deal with that alone: you can talk to Samaritans any time on 116 123 (free), or call 999 if you might not be safe. The readings below are only about the message, not about you.";

export const SCAM_NOTE =
  "This message matches a common scam pattern. Do not use any link, pay anything, or share card details, passwords or codes. To check, contact the organisation or person through a number or website you already know, not one in the message.";

export const BOUNDARY_NOTE =
  "You have already said no. You can keep saying no, and you do not have to give more reasons.";

const DISTRESS = [
  /\b(kill|hurt|harm|cut)(ing)? my ?self\b/,
  /\bsuicid/,
  /\bend(ing)? (it all|my life|it)\b/,
  /\b(want|wanted|wish) (to|i could) (die|disappear|vanish|not exist|not be here|not wake up)/,
  /\b(don'?t|do not) want to (be here|exist|live|wake up|be alive)/,
  /\b(better off|be better) (without me|if i (was|were) (gone|dead|not here)|if i (just )?(disappeared|died|wasn'?t here))/,
  /\bno (point|reason) (in )?(living|being alive|going on|anything|to live)\b/,
  /\b(can'?t|cannot) (go on|do this anymore|take (it|this) anymore|cope anymore)\b/,
  /\bi('?m| am) (a )?(burden|worthless|useless at everything)\b/,
  /\b(feel|feeling) (like )?(a burden|completely worthless|worthless)\b/,
  /\bi hate (my ?self|being alive)\b/,
  /\b(just )?want (it|everything|it all) to stop\b/,
  /\bnot (want|wanting) to (be|stay) alive\b/,
  /\boverdose\b/,
  /\bnothing (i do |i say )?matters\b/,
  /\bi (feel|'?m feeling|am feeling) (very |so |really |completely )?(hopeless|empty inside|like giving up)\b/,
  /\bi('?m| am) (feeling )?(so|very|really) low\b/,
];

// Scam: a request for money, codes or credentials, or a link, combined with a
// pressure or impersonation cue. Two-part rule to avoid flagging genuine
// delivery or bank notices that ask for nothing.
const SCAM_ASK = [
  /\b(card|bank) (details|number)\b/,
  /\b(security|verification|one[- ]time|6[- ]digit|confirmation) code\b/,
  /\b(read|send|tell|give) (me |us )?(the |that )?code\b/,
  /\bpin\b/,
  /\bpassword\b/,
  /\b(pay|paying|payment of|unpaid|fee of|redelivery fee|small fee|processing fee)\b/,
  /\bgift ?cards?\b/,
  /\b(bitcoin|crypto|usdt|wallet address)\b/,
  /\b(transfer|send|wire) (me |us )?(the |some )?(money|funds|£|\$|\d)/,
  /\b(https?:\/\/|www\.|[a-z0-9-]+\.(info|xyz|top|click|link|co|com|net|uk|io)\/\S*)/,
  /\bclick (the|this) link\b/,
  /\b(verify|confirm|update|unlock) (your )?(account|identity|details|payment)/,
  /\b(safe|secure|holding) account\b/,
  /\b(need|send|lend|borrow|wire)\b[^.]{0,30}(\d[\d,.]* ?(pounds|dollars|quid|euros|grand)|[£$€] ?\d)/,
];
const SCAM_CUE = [
  /\bwithin (\d+|twenty[- ]four|24|12|48) (hours|hrs|minutes)\b/,
  /\b(today|immediately|urgent(ly)?|right now|asap|final (notice|reminder)|last chance)\b/,
  /\b(will be|has been|is) (suspended|locked|cancelled|canceled|returned|closed|frozen|blocked)\b/,
  /\b(hmrc|irs|tax refund|refund of|you are owed|you('?re| are) owed)\b/,
  /\b(your bank|fraud (team|department)|from the bank|security team)\b/,
  /\b(new number|lost my phone|broke my phone|this is my new)\b/,
  /\b(won|winner|prize|claim your|selected)\b/,
  /\b(parcel|package|delivery)\b.*\b(fee|pay|unpaid)\b|\b(fee|pay|unpaid)\b.*\b(parcel|package|delivery)\b/,
  /\b(guaranteed|returns?|profit|double your)\b/,
  /\b(don'?t tell|keep this between|do not tell)\b/,
  /\b(only one i trust|never felt like this|never met|oil rig|stuck abroad)\b/,
  /\b(move|transfer) (it|your money|your savings)\b/,
];

// Boundary: the user reports a no, and the other person's message pushes.
const SAID_NO = [
  /\bi (said|told (him|her|them|you)|already said|made it clear|explained) (that )?(no|i (can'?t|won'?t|don'?t want|wasn'?t|am not|'?m not|would rather not|need))/,
  /\bi (said|told (him|her|them)) no\b/,
  /\bi (already )?(declined|refused|turned (it|him|her|them) down|said i('?d| would) rather not)\b/,
  /\bi (don'?t|do not|didn'?t) want to\b/,
  /\bi('?ve| have) (already )?said no\b/,
  /\bi (said|told (him|her|them)) (that )?i('?m| am)? ?(can'?t|cannot|not able|unable|won'?t|don'?t want|do not want)/,
  /\bi('?m| am) not able to\b/,
  /\b(don'?t|do not) want to see (him|her|them) again\b/,
  /\basked (me )?(two|three|four|several|many|\d+) times\b/,
];
const PUSH = [
  /\b(come on|just this once|only (a|one|two|\d+)|please just|you('?d| would) do it if|if you (really|actually) (cared|loved|were))/,
  /\b(i'?ll|i will) (ask|keep asking|call|come) (again|round|anyway)\b/,
  /\b(why not|what'?s the (big )?deal|don'?t be (like that|silly|boring|selfish|difficult))\b/,
  /\b(you owe me|after (everything|all) i('?ve)? (done|did))\b/,
  /\b(real (mate|friend)|true friend|good (son|daughter|friend|partner))\b/,
  /\b(not taking no|won'?t take no|no isn'?t an (answer|option))\b/,
  /\b(you always|you never|everyone else (is|will|has))\b/,
  /\b(i('?ve)? (already )?(booked|told everyone|bought|put you down))\b/,
  /\bi('?ve| have) already (told|said|promised)\b/,
  /\b(can|could) you not just\b/,
  /\byou'?ll change your mind\b|\byou'?re overthinking\b/,
  /\b(one more|last time i'?m asking|i'?ll pick you up)\b/,
  /\b(counting on you|let (us|me|the family|everyone) down)\b/,
  /\bi really need (this|you to)\b/,
];

const any = (pats: RegExp[], s: string) => pats.some((p) => p.test(s));

export function safetyCheck(input: SafetyInput): Safety {
  const self = [input.context, input.own, input.about].filter(Boolean).join(" ").toLowerCase();
  const msg = input.text.toLowerCase();
  const out: Safety = {};
  if (any(DISTRESS, self)) out.distress = DISTRESS_NOTE;
  if (input.mode === "them") {
    // The cue may be in the situation ("someone called saying they are from my bank").
    if (any(SCAM_ASK, msg) && any(SCAM_CUE, msg + " " + self)) out.scam = SCAM_NOTE;
    if (any(SAID_NO, self) && any(PUSH, msg)) out.boundary = BOUNDARY_NOTE;
  }
  return out;
}

// Reply ideas that comply with a scam. Dropped when the scam flag is set.
const COMPLY = [
  /\b(i'?ll|i will|let me|i can|ok,? i'?ll) (pay|send|confirm|click|transfer|read|give|share|enter|update)\b/,
  /\bhere (is|are) (my|the) (card|code|details|pin)\b/,
];

export function safeReplies<T extends { reply: string }>(replies: T[], safety: Safety): T[] {
  if (!safety.scam) return replies;
  return replies.filter((r) => !any(COMPLY, r.reply.toLowerCase()));
}
