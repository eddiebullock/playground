// Builds the two Word documents for collaborators in this folder (2026-11-24):
//   VLM_face_cues_results_summary.docx       the results, with Figures 1 to 6 of the paper
//   VLM_face_cues_how_the_study_worked.docx  the design, tests and safeguards, in plain language
// Every number is copied by hand from the locked results (research/paper/v2/*.md and
// research/RESULTS_REGISTRY.md); nothing is computed here. The figures are the committed ones in
// research/paper/v2/figures, which hold no face images.
// Needs the npm package docx 9.5.0, for example:
//   npm install --prefix /tmp/docxjs docx@9.5.0
//   NODE_PATH=/tmp/docxjs/node_modules node reports/build_collaborator_docs.js
// After a build, the zip's folder entries were removed with: zip -d FILE '*/'
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType,
  ShadingType, AlignmentType, ImageRun, BorderStyle, Footer, PageNumber, LevelFormat,
  VerticalAlign, TableLayoutType,
} = require('docx');

const FIG = path.join(__dirname, '..', 'research', 'paper', 'v2', 'figures');
const OUTDIR = process.argv[2] || __dirname;
const FONT = 'Calibri';
const C = { ink: '1B2633', muted: '56657A', accent: '1F4E79', rule: 'C5CFD9', head: 'E3EAF2', callout: 'EDF2F8', soft: 'F5F7FA' };
const TEXT_W = 9026; // A4 (11906) minus two 1440 margins

// ---- text helpers ---------------------------------------------------------------------
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0;
  let m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ ...base, text: text.slice(last, m.index) }));
    const tok = m[0];
    if (tok.startsWith('**')) out.push(new TextRun({ ...base, text: tok.slice(2, -2), bold: true }));
    else out.push(new TextRun({ ...base, text: tok.slice(1, -1), italics: true }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ ...base, text: text.slice(last) }));
  return out;
}
const p = (text, extra = {}) => new Paragraph({ children: runs(text), spacing: { after: 160, line: 276 }, ...extra });
const h1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
const h2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
const note = (text) => new Paragraph({ children: runs(text, { size: 18, color: C.muted }), spacing: { before: 80, after: 200, line: 252 } });
const lead = (text) => new Paragraph({ children: runs(text), keepNext: true, spacing: { before: 120, after: 80 } });
const bullets = (items) => items.map((t) => new Paragraph({ numbering: { reference: 'bullets', level: 0 }, children: runs(t), spacing: { after: 90, line: 276 } }));
const numbers = (items, instance) => items.map((t) => new Paragraph({ numbering: { reference: 'numbers', level: 0, instance }, children: runs(t), spacing: { after: 90, line: 276 } }));
const callout = (items) => items.map((t, i) => new Paragraph({
  children: runs(t),
  shading: { type: ShadingType.CLEAR, color: 'auto', fill: C.callout },
  border: { left: { style: BorderStyle.SINGLE, size: 24, color: C.accent, space: 10 } },
  indent: { left: 240, right: 120 },
  spacing: { before: i === 0 ? 120 : 0, after: i === items.length - 1 ? 240 : 100, line: 288 },
}));

function table(headers, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  if (total !== TEXT_W) throw new Error(`table widths sum to ${total}, not ${TEXT_W}`);
  const line = { style: BorderStyle.SINGLE, size: 4, color: C.rule };
  const borders = { top: line, bottom: line, left: line, right: line };
  const size = opts.size || 18;
  const cell = (text, w, head, shade) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    borders,
    shading: head ? { type: ShadingType.CLEAR, color: 'auto', fill: C.head }
      : shade ? { type: ShadingType.CLEAR, color: 'auto', fill: shade } : undefined,
    margins: { top: 70, bottom: 70, left: 110, right: 110 },
    verticalAlign: VerticalAlign.TOP,
    children: [new Paragraph({ children: runs(String(text), head ? { size, bold: true } : { size }), spacing: { after: 0, line: 252 } })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    layout: TableLayoutType.FIXED,
    rows: [
      new TableRow({ tableHeader: true, cantSplit: true, children: headers.map((h, i) => cell(h, widths[i], true)) }),
      ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, widths[i], false, opts.firstCol && i === 0 ? C.soft : undefined)) })),
    ],
  });
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function figure(name, caption, widthPx) {
  const file = path.join(FIG, name);
  const { w, h } = pngSize(file);
  const title = caption.split('. ')[0];
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 80 },
      children: [new ImageRun({ type: 'png', data: fs.readFileSync(file), transformation: { width: widthPx, height: Math.round((widthPx * h) / w) }, altText: { title, description: caption, name } })],
    }),
    new Paragraph({ children: runs(caption, { size: 18, color: C.muted }), spacing: { after: 240, line: 252 } }),
  ];
}

function titleBlock(title, subtitle, meta) {
  return [
    new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun(title)] }),
    new Paragraph({ children: [new TextRun({ text: subtitle, size: 28, color: C.muted })], spacing: { after: 100 } }),
    new Paragraph({ children: runs(meta, { size: 19, color: C.muted }), spacing: { after: 300 }, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: C.rule, space: 8 } } }),
  ];
}

function makeDoc(title, footerText, children) {
  return new Document({
    title,
    styles: {
      default: {
        document: { run: { font: FONT, size: 22, color: C.ink }, paragraph: { spacing: { after: 120, line: 276 } } },
        title: { run: { font: FONT, size: 44, bold: true, color: C.accent }, paragraph: { spacing: { before: 0, after: 120 }, keepNext: true } },
        heading1: { run: { font: FONT, size: 30, bold: true, color: C.accent }, paragraph: { spacing: { before: 420, after: 140 }, keepNext: true, outlineLevel: 0 } },
        heading2: { run: { font: FONT, size: 24, bold: true, color: C.ink }, paragraph: { spacing: { before: 300, after: 100 }, keepNext: true, outlineLevel: 1 } },
      },
    },
    numbering: {
      config: [
        { reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 420, hanging: 280 } } } }] },
        { reference: 'numbers', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 420, hanging: 320 } } } }] },
      ],
    },
    sections: [{
      properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1300, bottom: 1300, left: 1440, right: 1440 } } },
      footers: {
        default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [`${footerText}   |   Page `, PageNumber.CURRENT, ' of ', PageNumber.TOTAL_PAGES], size: 16, color: C.muted })] })] }),
      },
      children,
    }],
  });
}

// ---- document 1: results summary -----------------------------------------------------
const results = [
  ...titleBlock(
    'Do vision-language models read faces the way people do?',
    'Results summary for discussion',
    'Draft, unpublished results. 24 November 2026. Working title of the paper: *Beyond matching answers: testing whether vision-language models use facial cues as people do*.',
  ),
  h1('Summary'),
  ...callout([
    '**Headline.** Matching people’s answers cannot show that a vision-language model reads faces as people do. When facial cues are randomised, one small open model, Gemma 4, shares people’s cue effects for basic emotions. This replicated in a second human sample from the same laboratory, though the agreement is lower than the agreement between two human observer groups. The other eligible model, Qwen3-VL, was not shown to share them.',
  ]),
  p('We tested three small open vision-language models, Gemma 4 E4B, Qwen3-VL-8B and Molmo2-O-7B, on the EU-Emotion face clips (243 silent clips, 21 labels, 17 actors) and on faces rendered from randomised facial-action recipes that people had judged in published reverse-correlation studies. Two larger models in the same families, Gemma 4 31B and Qwen3-VL-32B, were added as a preregistered extension. No new human data were collected. Every confirmatory test was fixed in a dated amendment before its data were analysed, and each could return a match, a mismatch or an inconclusive result. The companion document, *How the study worked*, explains the design step by step.'),
  lead('**Five findings**'),
  ...numbers([
    '**Accuracy alone cannot show that a model used the face.** Our earlier design, and several published ones, could not rule out that option order or prompt text supplied the answer.',
    '**All three models’ answers depend on the face** once option order is rotated: swapping the face cut accuracy by 7 to 40 points. One model, Molmo2, never exceeded chance, so dependence is not competence.',
    '**Agreement with people on natural faces cannot identify which cues a model uses.** We show this formally, and all three models aligned with the human response pattern on natural clips, including the one at chance.',
    '**With cues randomised, Gemma 4’s cue effects align with people’s for basic emotions:** residual alignment of .386 and .516 in two observer groups and .483 in a second sample, against a bound of .30, but below the .780 at which two human groups agree. Qwen3-VL was inconclusive, and not aligned on the second sample’s renders.',
    '**Supporting tests agree within their scope.** Masks and filters moved judgements as human studies predict on about half the single-face contrasts, and evidence from the face moved judgements beyond what emotion words alone supply.',
  ], 1),

  h1('Results at a glance'),
  p('Each cell gives the class or value from the locked analysis, and models are never pooled. Face dependence is the accuracy lost when the face is swapped. Molmo2 failed two vision gates, so it was not eligible for the cue-level tests.'),
  table(['Test', 'Gemma 4', 'Qwen3-VL', 'Molmo2'], [
    ['Face dependence, composite input', '.228, passes', '.402, passes', '.071, passes (accuracy at chance)'],
    ['Face dependence, single face crop', '.216, passes', '.244, passes', '.063, passes (accuracy below chance)'],
    ['Natural clips: alignment with the human response pattern', '.418, aligned', '.549, aligned', '.543, aligned'],
    ['Randomised cues, basic emotions', 'Aligned in both groups (.386 WC, .516 EA)', 'Inconclusive (a gate failed on WC; EA .264, interval spans the bound)', 'Not eligible'],
    ['Randomised cues, conversational signals', 'Mixed (aligned on EA, .369; not aligned on these renders on WC, .025)', 'Inconclusive (gates failed)', 'Not eligible'],
    ['Second human sample, basic emotions', 'Aligned, .483: replicated (same lab and generator)', 'Not aligned on these renders, .230; replication not testable', 'Not eligible'],
    ['Masks and filters, single face (10 contrasts)', '5 match the prediction, 5 inconclusive', '4 match, 6 inconclusive', 'Not eligible'],
    ['Frame-order edits, video (6 contrasts)', '1 opposes (not robust), 5 inconclusive', '6 inconclusive', 'Not eligible'],
    ['Face evidence beyond word knowledge', 'Shown', 'Shown', 'Not eligible'],
  ], [2526, 2200, 2300, 2000], { firstCol: true }),
  note('Larger models: on a single face crop, Gemma 4 31B (.316) and Qwen3-VL-32B (.329) also depend on the face. Neither has a validity-gate result, so by the preregistered stop rule their other tests did not run. WC, Western Caucasian observers; EA, East Asian observers.'),

  h1('Findings'),
  h2('1. Accuracy alone cannot show that a model used the face'),
  p('A multiple-choice score measures face reading only if the answer cannot be found without the face. Our earlier design failed on two counts, and we withdrew it before any of the work reported here:'),
  ...bullets([
    'The acted label was the first option on 243 of 243 items, so accuracy equals the share of first-option choices.',
    'Each of the 21 labels had its own option set, so the options identify the category without the image. Category alone correlated .503 with human response entropy.',
  ]),
  p('We then coded eight recent published studies of emotion or mental-state reading by vision-language models. Target position was controlled in one and partly in one. None estimated the accuracy lost when the face is swapped under a fixed prompt: five had no image-free, mismatched or swapped-face control, and three had a partial one.'),
  p('Two more conditions must hold. The face has to reach the model: on Gemma 4, the composite frame grid of the earlier design left a median of 1.08 image tokens across the face, against 9.01 on a single face crop and a gate of 8. And the target has to be fair: for items acted as Kind, Unfriendly or Jealous, face-only human observers chose the acted label little more often than chance (.229 to .257, against 1/6). Three vision gates checked that each model’s output depends on its image. Gemma 4 and Qwen3-VL passed, and Molmo2 failed two.'),
  ...figure('fig1.png', 'Figure 1. Validity audit. (a) Position of the acted label in the earlier prompt. (b) One option set per label. (c) Image tokens across the face per input and model; dashed line, the gate of 8. (d) Weak labels in the human norms; dashed line, chance. (e) Eight published designs, as coded from the papers.', 540),

  h2('2. Answers depend on the face once option order is controlled'),
  p('Over five runs, the acted label was rotated through every position and the prompt was held fixed. The face shown was either the item’s own or one from another clip, with another label and another actor. By the preregistered rule, a model’s answers depend on the face if accuracy falls reliably when the face is swapped: an actor-level test, Holm-corrected across the three models, with a bootstrap interval above zero.'),
  table(['Model', 'Accuracy, own face', 'Accuracy, swapped face', 'Loss (95% CI)', 'Loss, single face crop'], [
    ['Qwen3-VL', '.469', '.067', '.402 (.350 to .461)', '.244'],
    ['Gemma 4', '.402', '.174', '.228 (.172 to .283)', '.216'],
    ['Molmo2', '.163', '.092', '.071 (.049 to .094)', '.063'],
  ], [1700, 1750, 1850, 2026, 1700], { firstCol: true }),
  note('Composite input unless stated; Holm *P* = .00009 for each model on the composite input. Chance is 1/6 (.167).'),
  p('**Dependence is not competence.** Molmo2’s accuracy with the item’s own face was at chance on the composite input and below chance on the single face crop (.072). It passes because swapped faces pull its accuracy down further; on the composite input it answered None of the above for 58% of prompts with the right face. For Qwen3-VL and Gemma 4, accuracy with the right face was well above chance, and it fell by 40 and 23 points when the face was swapped.'),
  ...figure('fig2.png', 'Figure 2. Face dependence. (a) Accuracy with the real face minus accuracy with a mismatched face, per model and input, with 95% intervals; filled markers meet the preregistered rule. (b) Accuracy with the real and the mismatched face; dashed line, chance. (c, d) Accuracy by position of the acted label, on the composite input and the single face.', 600),

  h2('3. Agreement on natural faces cannot identify which cues a model uses'),
  p('Face dependence shows that an answer uses the image, not which facial information it uses. The usual next step compares a model’s responses with people’s on natural faces. We show formally that this cannot settle cue use. If two systems give the same responses on every natural item, every behavioural measure on those items is equal, yet their cue effects can take any pattern (Theorem 1). The reason is that facial movements co-occur on natural faces. For linear observers, how far behaviour constrains cue effects depends on how strongly the cues are correlated (Proposition 3).'),
  p('The natural clips are poorly conditioned. On the 243 EU-Emotion clips, the smallest eigenvalue of the correlation of 15 detector-measured cues was .054, with a condition number of 76.1, against .44 to .47 and 3.4 to 3.5 for the randomised human design. Uncorrelated cues would give 1 for both.'),
  p('The data show the limit in practice. Our preregistered natural-clip test regresses each system’s graded responses on four detector-measured families of facial actions and correlates the model’s pattern with the human one. All three models were aligned with the human pattern: .549 for Qwen3-VL, .418 for Gemma 4 and .543 for Molmo2 (permutation *P* = .0002 to .0006). That includes Molmo2, whose accuracy was at chance, and two models for which the face spanned only one or two image tokens on this input. The score supports alignment with the human pattern, not use of the same cues.'),
  ...figure('fig3.png', 'Figure 3. Why agreement is not enough. (a) Two observers that use different cues. (b) When the cues correlate, the observers’ outputs correlate as strongly, while their cue effects stay unrelated. (c) Systems built to match a fixed system on every natural item can reach any requested cue alignment with it. (d, e) The natural clips are far less well conditioned than the randomised designs.', 600),
  ...figure('fig4.png', 'Figure 4. Natural-clip pattern score per model (large markers) and for each face detector alone (small triangles), against the null band and the range reached by synthetic human-like and prior-only observers (grey bars).', 500),

  h2('4. With cues randomised, Gemma 4’s cue effects align with people’s for basic emotions'),
  p('In reverse-correlation experiments, observers judge animated faces whose facial action units (AUs) are set at random, which identifies the effect of each AU on human judgement. We used the public data of Snoek and colleagues, in which basic emotions and conversational signals were each judged by a Western Caucasian (WC) and an East Asian (EA) observer group. We rendered the same AU configurations as static faces at their peak and measured each model’s effect of every AU family on its responses. The models saw the same configurations, not the same stimuli people saw.'),
  p('We compared model and human cue effects by residual Cue Effect Alignment (CEA): the cosine between the two matrices of cue effects after removing each matrix’s own average effect, so that the score reflects which action moves which label. A model is **aligned** if it passes four validity gates, the rendered AUs move its responses, and residual CEA is significant after Holm correction, at least .30 and with a 95% interval above 0. It is **not aligned on these renders** if the reliability-corrected upper bound is below .30, and **inconclusive** otherwise.'),
  table(['Dataset', 'Gemma 4', 'Qwen3-VL'], [
    ['Basic emotions, WC', 'Aligned: .386 (.372 to .397)', 'Inconclusive: validity gate failed (domain limit)'],
    ['Basic emotions, EA', 'Aligned: .516 (.497 to .535)', 'Inconclusive: .264 (.219 to .299); corrected upper bound .306'],
    ['Conversational signals, WC', 'Not aligned on these renders: .025; corrected upper bound .053', 'Inconclusive: two validity gates failed'],
    ['Conversational signals, EA', 'Aligned: .369 (.348 to .385)', 'Inconclusive: validity gate failed (domain limit)'],
    ['Second human sample (Liu et al.), basic emotions, Western observers', 'Aligned: .483 (.469 to .496); replicated (same lab and generator)', 'Not aligned on these renders: .230 (.189 to .269); corrected upper bound .273'],
  ], [2526, 3250, 3250], { firstCol: true }),
  note('Residual CEA with 95% intervals. For comparison, the two human groups agree at .780 on basic emotions and .732 on conversational signals, and the two human samples at .690. No model interval reaches these, so no result is human-level by that standard.'),
  p('Conversational EA is the least secure of Gemma 4’s aligned results: restricted to the four AU families that a face detector clearly confirmed on the renders (AU01, 05, 25 and 26), it was .158. Because only those four were clearly confirmed, the alignment is with the rendered AU recipes. On the second sample, Qwen3-VL’s cue effects lined up with people’s above chance (Holm *P* = .037), but by less than the smallest value of interest.'),
  ...figure('fig5.png', 'Figure 5. Randomised cue test. (a) Residual CEA per model and dataset, with 95% intervals; black bars, the human between-group reference; dashed line, the bound of .30. (b) Raw CEA, a secondary measure. (c) Correlation of each model’s answers with the mean human answer on the same faces, descriptive.', 600),

  h2('5. Answers on the same faces track people’s only weakly'),
  p('Comparing answers on the same rendered faces gives a weaker picture (Figure 5c). Where Gemma 4’s cue effects were aligned, its responses correlated with the mean human response at only r = .140, .229 and .138, each less than half its residual CEA, and its most likely answer matched the most common human answer on 10% to 15% of stimuli. Qwen3-VL’s correlations ran from −.025 to .084. Judged by its answers alone, Gemma 4 would have looked only weakly similar to people where its cue effects met the alignment bound.'),
  p('This comparison is descriptive and has no noise ceiling. Each stimulus’s human response rests on far fewer trials than each cue effect, so the gap may partly or wholly reflect noise in the human responses. A split-half noise ceiling would settle it, and it is the one optional analysis still open.'),

  h2('6. Masks and filters move judgements as predicted on about half the single-face contrasts'),
  p('We edited the clips and asked whether each model’s option-free judgement moved as predictions from human studies say it should. The option-free judgement is read from the model’s probabilities for 106 emotion words when it is asked what the person feels, with no options offered. The predictions concern basic emotions only and come from other stimuli, so a match is agreement with a stated prediction, not with human data on these clips.'),
  ...bullets([
    'On a single face, region masks and spatial-frequency filters moved the judgement as predicted on 5 of 10 contrasts for Gemma 4 and on 4 of 10 for Qwen3-VL. The rest were inconclusive.',
    'Three matches per model also hold at the stricter *P* < .01: the mouth mask on Happy and the mouth-minus-eye contrast for both models, the low- minus high-pass filter on Sad for Gemma 4, and the brow mask on Angry for Qwen3-VL.',
    'Only the mouth mask on Happy, for both models, and the brow mask on Angry, for Qwen3-VL, held under every sensitivity analysis.',
    'On video, neither shuffled frames nor clips cut to their first or last third matched a prediction. For Gemma 4, shuffling frames on Happy and Surprised raised the acted label, against the prediction (*P* = .024, not robust). Every other video contrast was inconclusive.',
  ]),
  ...figure('fig6.png', 'Figure 6. Stimulus edits. Change in the option-free judgement in the predicted direction, with actor 95% intervals, per contrast, model and input; marker colour and fill give the decision class.', 600),

  h2('7. Face evidence, not word knowledge alone, moves the judgement'),
  p('A language model could answer from what an emotion word usually goes with rather than from the face. A preregistered prior-and-evidence test, with image-free, grey-image and mismatched-face baselines, separates the two. On both inputs, both models’ evidence class was Shown: evidence from the face moved each model’s judgement towards the portrayed emotion (estimates of .670 to 1.048, every 95% interval above 0, step-down *P* at most .00002). The face also moved each judgement towards people’s response distribution beyond the default (.084 to .153, *P* at most .00003).'),

  h2('8. Larger models in the same families'),
  p('As preregistered, Qwen3-VL-32B and Gemma 4 31B entered the same frozen tests. On a single face crop both depended on the face: the loss was .329 for Qwen3-VL-32B and .316 for Gemma 4 31B (Holm *P* = .00006 for each, over two models, a less strict family than the three-model one). Neither has a validity-gate result. The frozen gate code cannot run on Gemma 4 31B, and the rerun of Qwen3-VL-32B’s gates was deferred past the December GPU shutdown. By the preregistered stop rule neither ran the other tests, and nothing is inferred from that. The result extends face dependence within each family to about 32 billion parameters.'),

  h2('9. What could not be tested'),
  ...bullets([
    'Whether a cue’s effect travels from the image tokens through the language model. The only complete cut of the vision path was the whole path, whose effect is zero by construction, so the literal test cannot fail.',
    'Whether people and models make the same errors for the same reasons. The planned test failed its own check before any data and was never run.',
  ]),

  h1('What it means'),
  p('At the level of answers, the evidence that these models read faces as people do is weak in two ways. Accuracy on tests like ours can be earned without the face, and even perfect agreement on natural faces cannot show which cues a model uses. Our data show both limits at work: all three models depended on the face, including one at chance, and all three aligned with the human pattern on natural clips.'),
  p('When the cues were randomised, the models came apart. Gemma 4 shares a pattern of cue effects with people for basic emotions, in two observer groups and a second sample. That is a shared pattern on these renders, not evidence of the same strategy, and it falls short of human-to-human agreement. For conversational signals the result was mixed, which supports no claim about culture. Qwen3-VL was not shown to share people’s cue effects.'),
  p('Two descriptive observations suggest that answer-level similarity can mislead in practice as well as in principle, though neither is a test. The natural-clip scores ranked Qwen3-VL above Gemma 4 (.549 against .418), while on the randomised renders Gemma 4 was aligned and Qwen3-VL was not. And Gemma 4’s answers tracked people’s only weakly where its cue effects were aligned.'),
  p('**Two lessons for the field.** Studies that compare models with people on multiple-choice tasks should report text-only baselines and the accuracy lost when the face is swapped. And a claim that a model reads faces as people do is a claim about cues, so it needs data in which cues vary independently. Randomised human cue effects make that test possible now.'),

  h1('Limits'),
  ...numbers([
    'The cue-level results hold on rendered avatars at a static peak frame, a new stimulus set built from the AU recipes that people saw in motion. Most action families on the renders are not confirmed by a detector, so alignment is with rendered recipes, and nothing here speaks to real faces.',
    'The East Asian sets use faces not based on East Asian people, and the second human sample shares the first’s lab, face generator, protocol and culture. It is a second sample, not a second method.',
    'The models are open and small, about 4 to 8 billion parameters. Molmo2 left the cue-level tests on its vision gates, Qwen3-VL’s randomised results were limited by validity gates, and the larger models completed only the first test.',
    'Face dependence is not competence, and pattern alignment on natural clips is not evidence of shared cues.',
    'The predictions for the stimulus edits come from other stimuli, and about half of those contrasts cannot reach a null by design.',
  ], 2),

  h1('How the work was done'),
  ...bullets([
    '**Preregistered.** Each confirmatory test was fixed in a dated amendment, committed to the public repository before its data were analysed, and each analysis ran once.',
    '**Every change recorded.** 105 dated decisions and deviations are listed in the supplement, and none moved a threshold.',
    '**Three-way outcomes.** Every test could return a match, a mismatch or an inconclusive result, and inconclusive results are reported as such, never as nulls.',
    '**Reproducible.** Code and derived outputs are public. One script rebuilds every CPU result from a clean checkout and checks the GPU outputs against pinned hashes, and 1,179 automated tests pass.',
  ]),
  p('The companion document, *How the study worked*, gives the design of each test in plain language.'),

  h1('Where the paper stands'),
  ...bullets([
    'The analyses for this paper are complete. The paper is drafted in Nature Human Behaviour’s Article format: abstract 150 of 150 words, main text 4,995 of 5,000, six figures and one table, with Methods and a supplement.',
    'Proposed plan: a presubmission enquiry to Nature Human Behaviour, then a full submission if invited. If declined: Nature Communications or Nature Machine Intelligence, with TMLR as the fallback.',
    'Still to do before submission: confirm the ethics statement with the institution, archive the code with a DOI, deposit the preregistration record, finalise the data-availability route for EU-Emotion, finalise the reference list, and complete the author list and statements.',
  ]),
  h2('Questions to discuss'),
  ...numbers([
    '**Title.** The working title is *Beyond matching answers: testing whether vision-language models use facial cues as people do*. Alternatives lead with the finding (“A vision-language model partly shares people’s facial cue effects for basic emotions”) or with the thesis (“Matching people’s answers cannot show that vision-language models read faces as people do”).',
    '**Noise ceiling.** Whether to run the optional split-half noise ceiling for the answers-against-cues comparison (finding 5). It needs CPU only and would be labelled exploratory.',
    '**Frontier models.** Whether to extend the randomised-cue test to one or two large commercial models. It is the strongest way to widen the claim, but it costs money and the dataset licences must allow it.',
    '**Journal and preprint.** The journal plan above, and whether to post a preprint at submission.',
  ], 3),
];

// ---- document 2: how the study worked ------------------------------------------------
const explainer = [
  ...titleBlock(
    'How the study worked',
    'A plain-language guide to the design, the tests and the safeguards',
    'Companion to the results summary. Draft, unpublished. 24 November 2026.',
  ),
  h1('The question'),
  p('Several reports say that vision-language models, AI systems that take an image or video and answer questions about it in words, match or beat people at recognising emotions and mental states from faces. Those reports compare answers. This study asks a deeper question: when a model and a person look at the same face, do they rely on the same facial cues?'),
  p('Schyns and colleagues describe three levels at which an AI system can resemble people: the same behaviour, then the same stimulus features, then the same computations on those features. Comparing scores reaches the first level at most. This study aims at the second: do the models respond to the same facial movements as people do?'),
  lead('**The study in four questions**'),
  table(['Step', 'Question', 'How we tested it', 'What we found'], [
    ['1', 'Can the test be passed without the face?', 'Audited our own task, eight published designs, and how much of the face reaches each model', 'Yes, in our earlier design. No published design measured the accuracy that needs the face'],
    ['2', 'Does the model’s answer use the face?', 'Rotated the options and swapped the face under a fixed prompt', 'Yes for all three models, including one at chance'],
    ['3', 'Do answers on natural faces show which cues are used?', 'A formal proof, and a test of the response pattern on natural clips', 'No. All three models matched the human pattern, even the one at chance'],
    ['4', 'Do the model’s cue effects match people’s?', 'Faces with randomised facial actions, compared with human reverse-correlation data', 'Yes for Gemma 4 on basic emotions. Qwen3-VL was not shown to'],
  ], [700, 2500, 3026, 2800]),
  note('Supporting tests: stimulus edits (masks, filters and frame order), a test of face evidence against word knowledge, and an extension to larger models.'),

  h1('Two problems with comparing answers'),
  h2('Problem 1: a score may not measure face reading at all'),
  p('Language models are sensitive to the order and wording of answer options. In two text models, the gap between the best and worst orderings of the options reached about 13 to 75 percentage points. If the right answer is always the first option, a model that always picks the first option scores perfectly without looking at the face. And if each emotion label always comes with its own set of alternatives, the options alone reveal the answer.'),
  p('Our own first design had both faults: the acted label was first on all 243 items, and each of the 21 labels had its own option set. An audit on 4 November 2026 found this, and we withdrew that design before any of the reported work. We then checked eight recent published studies and found none that measured how much accuracy is lost when the face is swapped while the prompt stays the same.'),
  p('**The fix.** Rotate the options so that the right answer takes every position, keep the prompt fixed, and compare accuracy with the item’s own face against accuracy with a face from a different clip. The difference is the part of the score that needs the face.'),
  h2('Problem 2: natural faces cannot reveal which cues are used'),
  p('On natural faces, facial movements co-occur: a genuine smile raises the cheeks as well as pulling the lip corners. So, on natural faces, a system that responds to one movement looks the same as a system that responds to another. The illustration shows the idea with two imaginary observers.'),
  table(['Face', 'Lip-corner pull (smile)', 'Cheek raise', 'Observer S (uses the smile)', 'Observer T (uses the cheek raise)'], [
    ['Natural face 1', 'On', 'On', 'Happy', 'Happy'],
    ['Natural face 2', 'Off', 'Off', 'Not happy', 'Not happy'],
    ['Randomised face 3', 'On', 'Off', 'Happy', 'Not happy'],
    ['Randomised face 4', 'Off', 'On', 'Not happy', 'Happy'],
  ], [1926, 1700, 1400, 2000, 2000], { firstCol: true }),
  note('Illustration, not data. On natural faces the two observers always agree, so no amount of natural data can tell them apart. Only faces in which the cues vary independently can.'),
  p('The paper proves the general version (Theorem 1): two systems can give identical answers on every natural face while their cue effects follow any pattern at all. For simple linear observers, how much answers constrain cue use depends on how correlated the cues are (Proposition 3). On the 243 EU-Emotion clips the measured cues are strongly correlated (the smallest eigenvalue of their correlation matrix is .054, where 1 would mean uncorrelated), so answers on these clips say little about cue use.'),
  p('**The fix.** Use faces whose cues are set at random, so that no cue can stand in for another. Human data of this kind already exist: in reverse-correlation studies, people judged animated faces whose facial action units were switched on at random.'),
  ...figure('fig3.png', 'Figure 3 of the paper. (a) Two observers that use different cues. (b) When the cues correlate, their answers correlate as strongly, while their cue effects stay unrelated. (c) Systems that match a reference on every natural item can have any cue alignment with it. (d, e) Cue correlation on the natural clips compared with the randomised designs.', 600),

  h1('What we used'),
  h2('Models'),
  table(['Model', 'Size', 'Role in the study'], [
    ['Gemma 4 E4B', 'About 4 billion parameters (effective)', 'Main model; all tests'],
    ['Qwen3-VL-8B', '8 billion', 'Main model; all tests'],
    ['Molmo2-O-7B', '7 billion', 'Main model; failed two vision gates, so face dependence and natural clips only'],
    ['Gemma 4 31B', '31 billion', 'Larger-model extension'],
    ['Qwen3-VL-32B', '33 billion', 'Larger-model extension'],
  ], [2300, 2700, 4026], { firstCol: true }),
  p('All five are open-weight models. They ran on A100 GPUs at bfloat16 precision, each scored through the same full forward pass, and models are never pooled.', { spacing: { before: 160, after: 120 } }),
  h2('Faces and human data'),
  table(['Source', 'What it is', 'Used for'], [
    ['EU-Emotion', '243 silent face clips of 21 labels (20 emotions and mental states plus neutral), acted by 17 actors, with published human response norms. A collaborator dataset, used with its creators’ permission', 'Face dependence, natural clips, stimulus edits, prior and evidence'],
    ['Snoek et al., reverse-correlation data', 'People judged animated faces whose facial actions were set at random. Western Caucasian and East Asian groups: 60 observers each for basic emotions and 20 each for conversational signals. Public', 'The human cue effects for the randomised test'],
    ['Liu et al. (2022), Experiment II', '20 Western observers sorted 1,200 random facial animations into six emotions or “other”. Public', 'The second human sample'],
    ['CREMA-D', '24 clips outside every analysis set', 'Only to choose the option-free readout'],
  ], [2200, 4426, 2400], { firstCol: true }),
  p('No new human data were collected and no participant was contacted. No clip, frame or face image appears in the paper or the public repository.', { spacing: { before: 160, after: 120 } }),

  h1('How a model was asked'),
  ...bullets([
    '**Inputs.** Faces were shown in three ways: a composite input built from the video at one frame per second, up to 16 frames (a frame grid for Gemma 4 and Molmo2, a list of frames for Qwen3-VL); a single face crop (one peak frame chosen from facial landmarks, cropped to the face and sent at 768 by 768 pixels); and the video frames themselves, for the frame-order edits.',
    '**Six-option task.** Each EU-Emotion item offered five named labels plus None of the above. Across five runs the five labels were rotated, so that the right answer took each position once.',
    '**Scoring.** In the first face-dependence test, each prompt was answered 20 times and the most common answer was scored. From the single-face test on, we read the model’s exact probability for each option instead.',
    '**Option-free judgement.** For the stimulus edits and the prior-and-evidence test, the model was asked “In one plain word, with no formatting or punctuation, what is this person feeling?”, and we read its probabilities for 106 emotion words in 21 families. No options are shown, so option order cannot matter. This readout was chosen in advance on 24 calibration clips outside every analysis set.',
  ]),

  h1('Checks before any test'),
  p('Each test had gates: checks that must pass before a result counts. A failed or missing gate makes the result inconclusive, never a pass or a fail.'),
  ...bullets([
    '**Delivery.** For each input and model, we counted the image tokens covering the face, with a gate of 8. Gemma 4’s composite grid gave about 1, against about 9 for a single face crop, which is why the face-dependence test was repeated on a single face.',
    '**Vision gates.** Three gates, called option independence, replay-and-cut and token swap, checked that information from the image reaches the model’s output as expected. Their tolerances were fixed before any model ran. Gemma 4 and Qwen3-VL passed. Molmo2 failed option independence and token swap, so it was not eligible for the cue-level tests.',
    '**Readout gate.** Before the option-free judgement was used on a model and input, a gate checked that it carries image signal and not just a preference for common words.',
    '**Randomised-test gates.** Four checks ran before any human comparison: a face detector sees the rendered movements grow with their strength; the model’s answers change when the face’s pixels are removed; the same configuration gives closer answers on two different faces than two different configurations do; and the model does not put almost all its answers on None of the above or on any one option.',
    '**Recovery checks.** Before any model ran, synthetic observers with known answers were passed through each analysis, to confirm that it returns the right verdict. The randomised test’s main statistic passed 66 of 66 checks. A simpler version, raw CEA, could be fooled in these checks, so it was made a secondary measure before any model data existed.',
  ]),

  h1('The tests, one by one'),
  h2('1. Face dependence'),
  p('**Question.** Does the model’s answer depend on the face, once option order is controlled?'),
  p('**What we did.** Each of the 243 items ran under five rotations of its labels. In the real condition the model saw the item’s own clip. In the mismatch condition it saw the same prompt with a partner clip of a different label and a different actor. Face dependence is accuracy with the real face minus accuracy with the mismatched face. The first test used the composite input, and a second repeated it on a single face crop.'),
  p('**Rule.** A model passes if a one-sided sign-flip test over actors (100,000 draws) is significant after Holm correction across the three models, and the bootstrap 95% interval over actors (10,000 draws) lies above zero. A pass is called face dependence, never “reads faces”.'),
  p('**Result.** All three models passed on both inputs, with losses of .063 to .402.'),
  h2('2. Response pattern on natural clips'),
  p('**Question.** Do the model’s graded responses on natural clips follow the same pattern over facial cues as people’s?'),
  p('**What we did.** Face detectors (LibreFace and OpenFace 3.0) measured four families of facial actions on each clip: brow raise (AU1 and AU2), brow lower and nose wrinkle (AU4 and AU9), smile (AU6 and AU12) and mouth open (AU25 and AU26). For people and for each model, the probability of each answer was regressed on these families, and the two patterns were correlated: the family pattern score. Its null distribution came from 4,999 shuffles of cues within category.'),
  p('**Rule.** Set before any model data. Recovery on synthetic observers gave power of .995 to 1.0 and a false-positive rate of at most .06, so the test is confirmatory. A positive score means “aligned with the human pattern”, never “uses the same cues”.'),
  p('**Result.** All three models aligned, including the one at chance.'),
  h2('3. Randomised cues'),
  p('**Question.** Do a model’s effects of randomised facial actions match people’s?'),
  p('**Human side.** From the reverse-correlation data of Snoek and colleagues, we took the effect of each of 25 AU families on the probability of each answer: the change when the action is switched on rather than off, with the other actions as they were. Basic emotions and conversational signals were judged by Western Caucasian and East Asian groups, which gives four datasets.'),
  p('**Model side.** The same AU configurations were rendered on avatar faces (MakeHuman meshes with FACSHuman action targets; frontal view, flat lighting, a static peak frame, 768 by 768 pixels), 11,761 unique images in all. Each model saw one face at a time, with the answer options people had in rotated order, and the same effects were estimated from its answers.'),
  p('**Statistic.** Residual Cue Effect Alignment (CEA): the cosine between the human and the model matrix of effects, after removing from each its own average effect, so that the score reflects which action moves which answer. It runs from −1 to 1, where 1 means the same pattern.'),
  p('**Rule.** Aligned needs every gate passed, a significant signal test (the rendered actions move the answers), alignment significant after Holm correction across the two models, residual CEA of at least .30 and a 95% interval above zero. Not aligned on these renders needs the upper end of the interval, corrected for the model’s reliability, to fall below .30. Anything else is inconclusive. A task’s headline needs the two observer groups to agree; one of each is called Mixed.'),
  p('**Result.** Gemma 4 aligned for basic emotions in both groups and was Mixed for conversational signals. Qwen3-VL was inconclusive on both tasks.'),
  h2('4. Second human sample'),
  p('**Question.** Does the randomised-cue result hold against a different group of human observers?'),
  p('**What we did.** We used Liu and colleagues’ Experiment II, in which 20 Western observers judged 1,200 random facial animations from the same laboratory and face generator. Each configuration was rendered on four faces, a design chosen by recovery checks before any model saw a render, and the same statistic and rule were applied. What counts as a replication, and the words used for it, were fixed before data.'),
  p('**Result.** Gemma 4 aligned again, so its basic-emotion result replicated on a second human sample (same lab and generator). Qwen3-VL was not aligned on these renders.'),
  h2('5. Answers on the same faces'),
  p('A descriptive comparison with no decision rule: the correlation between each model’s answers and the mean human answer on the same rendered faces, with option means removed. It shows what an answer-level comparison would have concluded on the same stimuli. It has no noise ceiling yet.'),
  h2('6. Stimulus edits'),
  p('**Question.** Do edits that change what people see move a model’s judgement the way human studies predict?'),
  p('**What we did.** Region masks (mouth, eyes and brow), spatial-frequency filters (low-pass against high-pass), shuffled frame order, and clips cut to their first or last third, applied in memory to the EU-Emotion clips. Fourteen predictions from human studies give 16 contrasts, all on basic emotions. The outcome is the change in the option-free judgement of the acted emotion, averaged per actor.'),
  p('**Rule.** Exact sign-flip tests over actors, with Westfall-Young step-down correction within each model and input. Each contrast is classed as Matches prediction, Opposes prediction, Null or Inconclusive. Null needs, among other conditions, an equivalence test and a positive control, and about half the contrasts cannot reach it by design. A simulation before data showed that the test can be liberal when effects are skewed, so every verdict is also reported at the stricter *P* < .01.'),
  p('**Result.** About half the single-face contrasts matched their prediction; no video contrast did.'),
  h2('7. Prior and evidence'),
  p('**Question.** Does evidence from the face move the judgement, or could knowledge of emotion words alone explain it?'),
  p('**What we did.** The option-free judgement on each clip was compared with image-free, grey-image and mismatched-face baselines.'),
  p('**Rule.** Evidence is Shown if the face-evidence estimate is significant (step-down *P* < .05, with its interval above zero) and the readout gate passed. The other classes are Unmapped, Absent and Unresolved.'),
  p('**Result.** Shown for both models on both inputs.'),
  h2('8. Larger models'),
  p('**Question.** Do the frozen tests give the same answers on larger models in the same families?'),
  p('**What we did.** Qwen3-VL-32B and Gemma 4 31B went through the same frozen code, with their own Holm families and a stop rule: a model without a vision-gate result does not run the later tests.'),
  p('**Result.** Both depend on the face on a single crop. Neither has a gate result: the frozen gate code cannot run on Gemma 4 31B, and Qwen3-VL-32B’s rerun was deferred past the December GPU shutdown. So the later tests did not run.'),
  h2('9. Two questions closed as limits'),
  ...bullets([
    '**The route through the model.** Whether a cue’s effect is carried from the image tokens through the language model. Cutting the whole vision path makes every image look the same, so the effect under that cut is zero by construction and the literal test cannot fail. No complete cut was found inside the language model, so the paper makes no claim about routes.',
    '**Shared errors.** Whether people and models make the same errors for the same reasons, on CREMA-D clips. The test’s own check before any data failed, because a frame meant to hide a cue still carried it on one detector, so the test was never frozen or run.',
  ]),

  h1('How the work was kept honest'),
  p('The study ran as a series of preregistered amendments. Each fixed a test’s design, decision rule and code before its data existed or were read, and was committed to the public repository, most with a named tag.'),
  table(['Amendment', 'Test', 'Frozen'], [
    ['v1', 'Original task and hypotheses (later withdrawn)', '30 Oct 2026'],
    ['A1, A3', 'Face dependence, composite input and single face', '4 Nov 2026'],
    ['Natural-clip rules', 'Natural-clip pattern score', '4 Nov 2026; reading rule added 7 Nov'],
    ['A2', 'Stimulus edits and the option-free readout gate', '7 Nov 2026'],
    ['A4', 'Randomised cue test', '7 Nov 2026'],
    ['A5', 'Prior and evidence', '8 Nov 2026'],
    ['A7', 'Route through the model', '8 Nov 2026'],
    ['A8', 'Larger models', '9 Nov 2026'],
    ['A9', 'Second human sample', '9 Nov 2026'],
    ['A6', 'Shared errors', 'Closed 10 Nov 2026, never run'],
  ], [2000, 4526, 2500], { firstCol: true }),
  new Paragraph({ children: [], spacing: { after: 60 } }),
  ...bullets([
    '**Nothing read before its analysis was locked.** Each analysis ran once, after a check that recorded the hash of every input.',
    '**Thresholds never moved.** Every decision and deviation is dated and disclosed: 105 entries in the supplementary table, and twelve disclosures in Methods for anything that departed from a clean order. For example, a video-decoder fault forced the first face-dependence runs to be discarded with no result read.',
    '**Independent audits.** Before the main analyses ran, an independent audit compared each amendment’s frozen code with its text. Its rulings made readings stricter and changed no threshold.',
    '**Three-way outcomes.** Every test can return a match, a mismatch or an inconclusive result, and an inconclusive result is never reported as a null.',
    '**Fixed wording.** Claims use agreed words that do not overstate: face dependent, never “reads faces”; aligned with the human pattern, never “uses the same cues”; a shared pattern of cue effects, never “the same strategy”; and “not aligned on these renders”, always in full.',
    '**Reproducible.** One script checks out the repository, reruns every CPU step from public inputs, compares each output byte for byte with the committed file, and checks the GPU outputs against pinned hashes. The latest clean-checkout test rebuilt 47 outputs and verified 159 more by hash, with none failing, and 1,179 automated tests pass.',
  ]),

  h1('Timeline'),
  table(['When', 'What happened'], [
    ['30 Oct 2026', 'First design (v1) preregistered after pilot data.'],
    ['4 Nov', 'An audit found the option-order and option-set faults. v1 was withdrawn, and the face-dependence tests were frozen.'],
    ['5 Nov', 'The single-face face-dependence analysis ran.'],
    ['7 to 9 Nov', 'The remaining amendments were frozen: stimulus edits, randomised cues, prior and evidence, route test, larger models and second sample.'],
    ['9 Nov', 'The composite-input face-dependence analysis ran once.'],
    ['10 Nov', 'The route test’s first stage ran. The shared-errors test was closed after its own check failed.'],
    ['14 Nov', 'The stimulus-edit, randomised-cue, second-sample and prior-and-evidence analyses each ran once and were locked.'],
    ['15 to 23 Nov', 'The larger models ran the first stage of tests. The extension ended after that stage on 23 Nov.'],
    ['21 Nov', 'The paper was reframed around its headline.'],
  ], [1800, 7226], { firstCol: true }),

  h1('Glossary'),
  table(['Term', 'Meaning'], [
    ['Vision-language model', 'An AI system that takes images or video together with text and answers in text.'],
    ['Image tokens', 'The pieces a model cuts an image into. Few tokens across the face means little facial detail reaches the model.'],
    ['Action unit (AU)', 'A single facial movement in the Facial Action Coding System. For example, AU12 is the lip-corner pull of a smile.'],
    ['Reverse correlation', 'A method in which people judge faces whose features are set at random, so that the effect of each feature on their judgements can be estimated.'],
    ['Cue effect', 'The change in the probability of an answer when one facial action is switched on rather than off, with the others unchanged.'],
    ['Residual Cue Effect Alignment (CEA)', 'The cosine similarity between a model’s and people’s matrices of cue effects, after removing each matrix’s average effect: 1 means the same pattern, 0 unrelated, −1 opposite.'],
    ['Bound of .30', 'The smallest alignment treated as meaningful, set before any data (the smallest effect size of interest).'],
    ['Face dependence', 'Accuracy with the item’s own face minus accuracy with a mismatched face, under the same prompt.'],
    ['Option-free judgement', 'The model’s probabilities over 106 emotion words when asked what the person feels, with no options shown.'],
    ['Validity gate', 'A check that must pass before a result counts. A failed or missing gate makes the result inconclusive.'],
    ['Recovery check', 'Running an analysis on synthetic observers whose answers are known, to confirm that it returns the right verdict.'],
    ['Holm correction', 'A standard adjustment of p-values when the same test is run on several models.'],
    ['Sign-flip test', 'A test that randomly flips the sign of each actor’s effect, to see how often a result this large arises by chance.'],
    ['Bootstrap interval', 'A confidence interval obtained by resampling actors or stimuli many times.'],
    ['Permutation test', 'A test that shuffles the pairing of the data to build the distribution expected by chance.'],
    ['Noise ceiling', 'The highest agreement achievable given the noise in the human data.'],
    ['WC and EA', 'The Western Caucasian and East Asian observer groups in the Snoek data.'],
    ['Preregistered amendment', 'A dated document, committed to the public repository, that fixes a test’s design, rule and code before its data are analysed.'],
    ['bfloat16', 'A reduced numerical precision used to run large models on GPUs.'],
  ], [2700, 6326], { firstCol: true }),
];

async function main() {
  fs.mkdirSync(OUTDIR, { recursive: true });
  const docs = [
    ['VLM_face_cues_results_summary.docx', makeDoc('Do vision-language models read faces the way people do?', 'Results summary  |  Draft, unpublished  |  24 November 2026', results)],
    ['VLM_face_cues_how_the_study_worked.docx', makeDoc('How the study worked', 'How the study worked  |  Draft, unpublished  |  24 November 2026', explainer)],
  ];
  for (const [name, doc] of docs) {
    const buf = await Packer.toBuffer(doc);
    fs.writeFileSync(path.join(OUTDIR, name), buf);
    console.log(`wrote ${name} (${buf.length.toLocaleString()} bytes)`);
  }
}
main().catch((e) => { console.error(e); process.exit(1); });
