# VLM versus human mental-state identification from faces: theory, datasets, venues

Notes written 2026-11-04. Evidence marks used throughout:
- [V] = I opened the primary record, or downloaded and inspected the file.
- [A] = abstract or official listing only, via arXiv, Europe PMC or a publisher record.
- [S] = search-tool summary only. The full text was not opened, so check before citing.

Full texts could not be fetched for Gershman 2024, the Snoek 2023 PDF, the PNAS Snoek paper, the Scientific Reports 2026 paper or the GEMEP page. Those items rest on abstracts or summaries. Local dataset counts below are my own computations on the public files.

## A1. What Gershman's recent work suggests for defining "how similar are these models to humans"

### Takeaway
Gershman's lab has no paper on faces or emotion perception. Its recent human-versus-model work still gives two templates: fit a set of named cognitive models to the LLM and to humans, and compare matched trials against human split-half reliability. The sharpest published definition of similarity is the Glasgow three-degree ladder (behaviour, then features, then computation), which maps directly onto the supervisor's request.

### Cited Findings
- A title search of Gershman's full publication list for emotion, facial, affect, face and multimodal returned no paper on face or emotion perception. His emotion-adjacent items are computational-psychiatry papers such as anhedonia and anxiety modelling. [V] — [Gershman lab publications](https://gershmanlab.com/pubs.html)
- Gershman 2024 argues that the historical story is more nuanced than "neuroscience drives AI": AI researchers were "loosely inspired by the brain, but ideas flowed mostly in the other direction". [A; full text not read] — [Biological Cybernetics, DOI 10.1007/s00422-024-00983-2](https://doi.org/10.1007/s00422-024-00983-2)
- CogGym (Gershman is a co-author, preprint, Sept 2026) converts 258 commonsense-reasoning experiments from 100 papers into a common markup and tests 50 LLMs against humans on matched trials. Best-model R² is 0.59 on text, 0.58 on image and 0.43 on video, against human split-half values of 0.93, 0.95 and 0.92. [A] — [arXiv 2609.21259](https://arxiv.org/abs/2609.21259). A separate project page reports lower figures (R² about 0.23 to 0.33 against 0.85) and says models fit the group mean but not the distribution. That page may be a different project or version, so treat those figures as unresolved. [S] — [project page](https://ningshan-gif.github.io/project/cogym)
- Xiang, Bigelow, Gerstenberg, Ullman and Gershman fitted seven cognitive models of responsibility and reward to LLMs and to humans. LLMs mainly used force (how much agents actually did). Humans valued actual and counterfactual effort. The conclusion names a strategy difference, not just an accuracy gap. [S] — [CogSci 2025](https://cicl.stanford.edu/publication/xiang2025how/). A workshop follow-up tested ten models and found the divergence appears asymmetrically, on collaboration failures rather than successes. [S] — [NeurIPS 2025 workshop](https://neurips.cc/virtual/2025/129714)
- Sohail et al. (Aug 2026 preprint) fitted cognitive computational models to LLMs and 251 humans in economic games, and present this as a formal way to compare intelligence across humans and machines. [A] — [arXiv 2608.26291](https://arxiv.org/abs/2608.26291)
- Bari and Gershman 2024 argue that parts of psychopathology are resource-rational, using two theories: rational inattention (capacity limit on perceptual channels) and policy compression (capacity limit on action channels). They explicitly cover neurodevelopmental disorders. [A] — [Behavioral Neuroscience, DOI 10.1037/bne0000600](https://doi.org/10.1037/bne0000600)
- Schyns, Snoek and Daube set out three embedded degrees of algorithmic similarity: equivalent behavioural or brain responses, then equivalent stimulus features processed, then equivalent computations. They say current evidence of algorithmic equivalence between brains and DNNs is limited. [S] — [TiCS 2022](https://eprints.gla.ac.uk/278852)
- Daube et al. ranked DNNs on deeper similarity of the functional features behind behaviour as well as on prediction, using a controllable 3D generative face model plus reverse correlation. DNN predictions of human behaviour improved where shape and texture features overlapped with those driving humans. [S] — [Patterns 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8515012)
- Han, Poggio and Cheung replaced brain recordings with networks of known architecture and found system-identification performance is variable and depends on factors unrelated to the true architecture, such as stimuli. [S] — [ICML 2023](https://proceedings.mlr.press/v202/han23d.html)
- Avitan and Golan fitted 20 vision models to about 4.5 million THINGS judgments and tried to recover the generating model from synthetic responses. Recovery stayed below 80% even with millions of simulated trials. [A] — [NeurIPS 2025, arXiv 2510.23321](https://arxiv.org/abs/2510.23321)

### Inferences
- A workable breakdown of the supervisor's question is five sub-questions, each with its own test:
  1. Item level: does the model match the human distribution over options, relative to a noise ceiling?
  2. Error level: does it err on the same items and towards the same wrong options?
  3. Feature use: do the same cues causally change the answer in both?
  4. Objective: does it infer a latent cause (appraisal, event, inner state), or map appearance to a label?
  5. Population: does it reproduce individual differences, or only the mean?
- The word "encoded" in the supervisor's note needs care. Being able to decode an AU from model activations shows information is present, not that the model uses it. Only intervention shows use. The project's activation-patching scaffold (local `PIVOT.md`) is the natural tool.
- The Avitan and Golan and Han results mean any claim of the form "model X is closest to humans" should ship with a model-recovery analysis.
- Best pitch to Gershman: an extension of the Xiang or CogGym logic to perception of mental states, a domain absent from his publication list, asking a named strategy question rather than a benchmark question. The resource-rational test (A3, prediction 5) links to his own psychiatry work.

### Gaps
- I could not read the full Gershman 2024 text. A university mirror was blocked as walled content and I did not retry. Any statement on Marr levels from that paper is unverified.
- Marr's original text and Gershman, Horvitz and Tenenbaum's computational rationality paper were not located this session. Resource-rational analysis was located only as a record ([Lieder and Griffiths, BBS](https://www.cambridge.org/core/product/586866D9AD1D1EA7A1EECE217D392F4A)).
- The publication list was processed by a summariser, and titles do not capture every topic. An email to the lab should still check for unpublished work.

## A2. Which frameworks give discriminating predictions between human and VLM strategies

### Takeaway
Inverse planning, Bayesian cue integration and generative-versus-discriminative contrasts all make testable predictions, but the generative-versus-discriminative axis should be treated as hybrid rather than binary. The most practical discriminators are controversial-stimulus selection, reliability-weighted integration and text-override tests.

### Cited Findings
- Houlihan et al. model emotion prediction by inverse planning over beliefs and preferences, then compute appraisals, and match human predictions for 20 emotions. Social preferences matter for nearly every emotion. Code and data are public. [S] — [Phil Trans A 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10239682); [code](https://awesome.ecosyste.ms/projects/github.com%2Fdaeh%2Fcomputed-appraisals)
- Baker, Saxe and Tenenbaum frame action understanding as inverse planning. [A] — [Cognition 2009](https://doi.org/10.1016/j.cognition.2009.07.005)
- Wu et al. address rational inference of beliefs and desires from emotional expressions. [A, located by title] — [Cognitive Science 2018](https://doi.org/10.1111/cogs.12548)
- Ong, Zaki and Goodman model emotion inference as Bayesian integration of facial and situational cues, with cues weighted by reliability. [S] — [Cognition 2015](https://cascoglab.psy.utexas.edu/desmond/pubs/2015_Affective_Cognition.pdf); [review and roadmap](https://doi.org/10.1111/tops.12371). A test with more naturalistic stimuli found mixed results: situation alone often beat the integration model, with exceptions for amusement and happiness. [S] — [Nature Communications 2024](https://link.springer.com/article/10.1038/s41467-024-46670-5)
- Yildirim et al. propose fast feedforward inversion of a 3D face graphics program, which explains behaviour including the hollow-face illusion and maps onto primate face-processing circuitry. [S] — [Science Advances 2020](https://dspace.mit.edu/handle/1721.1/138360)
- Golan, Raju and Kriegeskorte use controversial stimuli, chosen so that models disagree, to separate models. Generative or hybrid models fitted human judgments better than pure discriminative DNNs, though none fully explained responses. [S] — [PNAS 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7703564)
- Guo et al. used controversial face pairs with 864 participants. Models trained on high-level invariant objectives (inverse rendering, identification or object classification) matched human judgments best, and models with different theories often tie on random faces. [A] — [arXiv 2605.12619](https://arxiv.org/abs/2605.12619)
- Peters et al. argue the generative-versus-discriminative dichotomy may be partly false and that the primate brain likely uses a hybrid. [A] — [arXiv 2401.06005](https://arxiv.org/abs/2401.06005). Chang Ortega et al. vary one mixing coefficient and report human alignment peaks at intermediate blends. [A, preprint] — [arXiv 2605.23819](https://arxiv.org/abs/2605.23819)
- Deng et al. report "blind faith in text": ten VLMs weight text over conflicting images, and open models show stronger text bias than proprietary ones. [S] — [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Deng_Words_or_Vision_Do_Vision-Language_Models_Have_Blind_Faith_in_CVPR_2025_paper.html)
- Brooks and Freeman show individual conceptual knowledge predicts the structure of facial emotion perception. [S] — [Nature Human Behaviour 2018](https://doi.org/10.1038/s41562-018-0376-6). A neuroimaging follow-up reports conceptual structure predicts neural structure over and above facial features. [S] — [PMC6689944](https://pmc.ncbi.nlm.nih.gov/articles/PMC6689944)
- Barrett et al. challenge inferring emotion from facial movements alone. [A, located by title] — [PSPI 2019](https://doi.org/10.1177/1529100619832930)
- LLM work finds strong appraisal behaviour. Yongsatianchot et al. report GPT-4 performs reverse appraisal well, while warning LLMs will not replace mechanistic models. [S] — [arXiv 2310.18322](https://arxiv.org/pdf/2310.18322). Bhattacharyya et al. argue appraisal predictions for others do not expose internal representations. [S] — [arXiv 2508.05880](https://arxiv.org/html/2508.05880v1)
- Prior VLM mental-state work is mainly score-level. GPT-4o beat humans on upright RMET and MRMET faces, was worse on inverted faces, showed a White-face advantage absent in humans, and made consistent errors. [A] — [Strachan et al., arXiv 2410.22309](https://arxiv.org/abs/2410.22309). A vision abstract reports smaller inversion loss than humans (about 30% against 50%). [S] — [JOV abstract](https://jov.arvojournals.org/article.aspx?articleid=2802000). Further: [Refoua et al. 2026, Scientific Reports](https://doi.org/10.1038/s41598-026-39292-y) [A] and [Akben et al., arXiv 2508.08830](https://arxiv.org/abs/2508.08830), which reports aggregated humans beating aggregated models [A].

### Inferences
These are my proposed predictions, not published results. Each states what would separate the theories.

| # | Contrast | Test | Label-association route predicts | Generative or human-like route predicts |
|---|---|---|---|---|
| 1 | Lexical prior vs perceptual evidence | Run each RMET or EU-Emotion item with the image, a blank and a scrambled image. Regress image-present log-odds on no-image log-odds plus option word features | No-image log-odds explain much item-level variance; synonym swaps change answers with the image fixed | Choices depend on the image; no-image baseline near flat; synonym swaps change little |
| 2 | Canonical stereotype vs perceiver mapping | Compare the model's AU-to-label mapping with published AU prototypes and with empirical human mappings from Snoek's data | Closer to textbook prototypes than to human perceivers, with weak culture accents | Reproduces human departures from prototypes and cultural differences |
| 3 | Reliability weighting | Degrade the face parametrically and add a conflicting verbal context | Fixed text weight regardless of image quality | Weight on the face falls smoothly with its reliability |
| 4 | Causal latent inference | Vary the stated event or appraisal, holding the face fixed; check whether inferred states change as inverse planning predicts, including counterfactual context | Label shifts follow word associations of the context | Shifts follow appraisal structure (expectation and preference) |

- Prediction 3 can be anchored to human data at no cost. CREMA-D gives per-clip vote distributions for voice-only, face-only and audiovisual conditions, so human cue combination can be tested clip by clip before touching a model.
- Prediction 2 suits controversial selection. Choose AU configurations where candidate mappings disagree, as in Guo et al., instead of averaging over all stimuli.
- Do not frame the result as "generative versus discriminative". A defensible claim is the weight of prior versus evidence, and whether that weight is calibrated to reliability.
- Temporal prediction: Jack et al. report early signals separate four categories and later ones six ([Current Biology 2014](https://doi.org/10.1016/j.cub.2013.11.064)) [S]. Frame shuffling on EU-Emotion should leave a frame-sampling VLM largely unchanged. This is a model-only test, because the public Snoek files hold AU amplitudes but no temporal parameters.

### Gaps
- I found no paper fitting an inverse-planning emotion model to a VLM on face stimuli. This is one search-based null, not proof of absence.
- Houlihan-style stimuli are game scenarios, not facial clips, so prediction 4 needs a new vignette design and has no matching EU-Emotion human data.
- Whether open VLMs on CSD3 take audio natively must be checked per model before using CREMA-D multimodally.

## A3. The two-route idea and computational psychiatry

### Takeaway
The human evidence for a verbal load on the RMET is strong enough to motivate a route analysis, but disputed enough that it should be a measured quantity, not an assumption. The most novel computational-psychiatry contribution is to fit one resource-rational choice model to CARD and open RMET item data and to VLMs.

### Cited Findings
- Peterson and Miller: in 42 students, verbal IQ contributed significantly to Eyes Test variance and a face-perception measure did not. [A] — [Frontiers 2012](https://doi.org/10.3389/fpsyg.2012.00220)
- Olderbak et al.: with 484 and 210 participants, the 10-item short form correlated with vocabulary at r = 0.49 and 0.62, more strongly than with emotion perception (0.34 in Study 1) or cognitive empathy. They urge reducing reliance on vocabulary. The Study 2 emotion-perception value is ambiguous in the paper. [V] — [Frontiers 2015](https://www.frontiersin.org/articles/10.3389/fpsyg.2015.01503/full)
- Oakley et al.: alexithymia, not autism, affected RMET scores in matched groups, while autistic participants were impaired on the MASC. They conclude the RMET indexes emotion recognition rather than ToM. A 2019 commentary contested parts of this. [S] — [J Abnormal Psychology 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4976760); [commentary](https://www.pubmed.gov/31580106); [reply](https://pure.royalholloway.ac.uk/files/34011659/OakleyResponseToCommentary.pdf)
- Higgins and colleagues raise construct-validity threats. [A, located by title] — [NBR 2023](https://doi.org/10.1016/j.neubiorev.2023.105088); [scoping review 2024](https://doi.org/10.1016/j.cpr.2023.102378)
- A 2020 study reports easy and positive items best separate autistic people from controls. [S] — [PMC7891586](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7891586/)
- Pellicano and Burr offer a Bayesian account of autistic perception. [Located by title] — [TiCS 2012](https://doi.org/10.1016/j.tics.2012.08.009). Van de Cruys et al. give a predictive-coding account. [Located by title] — [Psychological Review 2014](https://doi.org/10.1037/a0037665). Lawson et al. report that autistic adults overestimate volatility. [Located by title] — [Nature Neuroscience 2017](https://doi.org/10.1038/nn.4615)
- Lawson, Rees and Friston state that posterior expectations depend on relative precision, so they cannot separate weak priors from strong sensory precision. Posterior confidence can: it should rise if sensory precision is not attenuated and fall if prior precision is lost. They call the evidence circumstantial. [V, fetched summary] — [Frontiers 2014](https://www.frontiersin.org/articles/10.3389/fnhum.2014.00302/full)
- Karvelis et al.: autistic traits predicted greater weighting of sensory information in Bayesian visual integration. [Located by title] — [eLife 2018](https://doi.org/10.7554/elife.34115)
- Diffusion modelling: Powell et al. found autistic adolescents had lower processing efficiency and lower caution on face recognition and gaze discrimination. [A] — [Neuropsychology 2019](https://doi.org/10.1037/neu0000524). Pirrone et al. found more caution and no drift difference on a non-face task. [S] — [White Rose copy](https://eprints.whiterose.ac.uk/107355/1/Perceptual%20Judgment%20in%20ASD%20final%20copy.pdf). These disagree on caution. I found no diffusion study of facial emotion recognition in autistic samples. [S]
- Computational psychiatry using networks: Takahashi et al. trained a hierarchical recurrent network to predict facial expression movies and altered intrinsic excitability as an ASD analogue, which reduced generalisation and emotion clustering. [A] — [Scientific Reports 2021](https://doi.org/10.1038/s41598-021-94067-x). A neurorobotics model found both raised and lowered sensory precision could produce rigidity. [S] — [PMC6317752](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6317752/)
- LLMs as artificial subjects: anxiety-inducing prompts raised exploration and bias. [S] — [Coda-Forno et al., arXiv 2304.11111](https://arxiv.org/abs/2304.11111v2). State anxiety in LLMs has also been assessed and manipulated. [Located by title; content not read] — [Ben-Zion et al., npj Digital Medicine 2025](https://doi.org/10.1038/s41746-025-01512-6)
- A 2026 in-silico study produced recall gradients across simulated cognitive impairment. [S] — [MDPI AI 2026](https://www.mdpi.com/2673-2688/7/2/69). A 2022 review describes synthetic lesion studies in digital twins, and contrasts older lesion models of psychosis with deep language networks. [S] — [Molecular Psychiatry 2022](https://www.nature.com/articles/s41380-022-01743-z)

### Inferences
- **Prediction 5, the resource-rational test.** Rate-limited choice gives p(a|s) proportional to p(a) exp(beta U(s,a)), so lower capacity should move answers towards the marginal popularity of each option, not towards uniform guessing. Estimate U from high scorers. Then ask whether low scorers in CARD or the open normative RMET data lie on the same one-parameter curve (a capacity difference), or need a different U (a knowledge or concept difference). Trace the same curve in a VLM by reducing visual gain. This uses the option-level RMET data and links to Bari and Gershman.
- **Prediction 6, two-route profile.** Define a route-dependence index per item from A2 prediction 1. If humans have a verbal route, items high on it in the model should be those where selection varies most with education in the normative data, or with vocabulary if CARD has a verbal measure. If the items are unrelated, the model's verbal route is not an analogue of the human one. Report that null, as it is informative.
- **Prediction 7, prior versus sensory analogue.** Lawson's confidence asymmetry can be tested in a VLM by raising visual feature gain against damping text-derived context, at activation level. Both may shift answers similarly, while confidence should move in opposite directions if the analogy holds. If you implement this as logit arithmetic, the result is true by construction and shows nothing.
- MRMET is a natural experiment on vocabulary: same construct, more accessible words. However the images differ, so it is not a paired test.
- Diffusion modelling needs human response times at item level. EU-Emotion aggregates do not provide these. The CREMA-D files include a `ttr` column that may hold response times; I did not confirm its meaning. VLMs have no native RT. Token-level proxies would be weak, so I would not base a claim on them.

### Gaps
- Rosenblau's social-learning work in autism was not found by the exact title I tried, so it is uncited here.
- I did not locate a paper proposing a VLM as a computational-psychiatry analogue for face perception. The closest are Takahashi (not a VLM) and the LLM anxiety work (not perception).
- I did not verify whether CARD holds alexithymia or vocabulary measures. This decides how far prediction 6 can go. Under Oakley's argument, autism status alone is not an adequate contrast.

## A4. How to do model "lesioning" honestly

### Takeaway
A lesion in a VLM can test a computational hypothesis about routes. It cannot model autistic people. The honest design is competing, dose-graded interventions scored on a preregistered pattern of item-level effects against human profiles, with the possibility that several lesions fit equally.

### Cited Findings
- The neurorobotics result above shows opposite precision changes yielding similar rigidity, so one behaviour can have several lesion causes. [S] — [PMC6317752](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6317752/)
- Mediation analysis treats model components as mediators between input and output. [S] — [Vig et al.](https://arxiv.org/pdf/2004.12265). It is compatible with ablation and interchange interventions. [S] — [Geiger et al., causal abstraction](https://arxiv.org/pdf/2301.04709). Mediator choice has trade-offs in granularity and in whether the unit is a causal bottleneck. [S] — [Quest for the Right Mediator](https://doi.org/10.1162/COLI.a.572). Standard activation patching can pick up spurious effects. [S] — [arXiv 2412.02893](https://arxiv.org/pdf/2412.02893)
- Psychometric studies of LLMs have been criticised for importing anthropomorphic assumptions. [S; ResearchGate only, non-peer-reviewed status unclear] — [Quiroga 2025](https://www.researchgate.net/publication/398499894_The_Anthropomorphic_Fallacy_in_AI_Research_A_Critical_Analysis_of_Psychometric_Testing_of_Large_Language_Models)

### Inferences
Suggested minimum standard:
1. State that the lesion instantiates a claim about a computation, not a person. Avoid phrases like "autistic model".
2. Preregister the mapping from hypothesis to intervention before seeing human profile fits.
3. Run at least two rival lesions per hypothesis, for example visual gain, text-context gain, feature-specific ablation for an AU-linked feature, and a resolution or patch-dropout control.
4. Report dose-response curves, not one setting.
5. Score specificity. A good lesion reproduces which items fall and which are spared, and does not simply lower accuracy everywhere. A random-weight-noise control sets the bar.
6. Add a recovery check: simulate from each lesion and verify the analysis recovers the lesion type, following Avitan and Golan.
7. Do not treat persona prompting as a lesion. Prompts such as "answer as an autistic person" measure text stereotypes, which is itself a good negative control.
8. Do not repurpose a fine-tune collapse as a lesion, because it is unspecific.

### Gaps
- I found no published standard for lesion validity in LLMs or VLMs. The list above is my synthesis.

## A5. Critiques of treating models as cognitive models

### Takeaway
The main critiques converge: matching behaviour is not evidence for mechanism, and metaphor can do unearned work. The defensible stance is to treat VLMs as systems whose strategies are measured, and to report divergences as findings.

### Cited Findings
- Bowers et al. argue DNN models account for almost no findings in psychology and that predictive success can be mediated by systems sharing little with biological vision. [S] — [BBS 2023](https://research-information.bris.ac.uk/en/publications/deep-problems-with-neural-network-models-of-human-vision/). Open commentary includes a reply on what models are for. [S] — [Aston record](https://research.aston.ac.uk/en/publications/clarifying-status-of-dnns-as-models-of-human-vision/)
- MindSet: Vision packages 30 psychological findings for DNN testing, and its authors say their tests are not a strong test of biological modelling. [S] — [arXiv 2404.05290](https://arxiv.org/abs/2404.05290)
- Shah and Varma set out assumptions and pitfalls of mapping pre-trained language models onto cognition (architecture, training data, scale, interpretability) and propose credibility criteria. [A] — [arXiv 2501.12651](https://arxiv.org/abs/2501.12651)
- Li et al. argue LLM understanding is framed through metaphor, creating an anthropomorphic loop, and propose studying the distinct logic of models. [A] — [Communications Psychology 2026](https://doi.org/10.1038/s44271-026-00508-6)
- Centaur, an LLM fine-tuned on trial-level psychology data, predicts held-out human behaviour. [S] — [Binz et al., Nature 2025](https://pubmed.ncbi.nlm.nih.gov/40604288/). A critique reports it does not follow misleading instructions as a true language understander would. [S] — [National Science Open](https://nso-journal.org/10.1360/nso/20250053)
- Fel et al. report 84 ImageNet DNNs trade accuracy against alignment with human strategies. [A] — [arXiv 2211.04533](https://arxiv.org/abs/2211.04533). A 2026 preprint reports the same pattern for Gestalt grouping: 34 of 45 models beat humans on accuracy and few match human-human error consistency. [S] — [arXiv 2608.10195](https://arxiv.org/pdf/2608.10195)

### Inferences
- The RMET itself carries construct risk (A3). A paper claiming a VLM "reads minds" inherits those threats, and superhuman RMET scores may reflect the test's lexical structure. This is an alternative explanation worth testing rather than conceding.
- Report Fel-style accuracy-alignment curves to pre-empt the reading that high accuracy implies human-like strategy.
- Peers will ask about training exposure: RMET items and answer keys may be online. Test for this using MRMET, unseen variants and the no-image baseline.

### Gaps
- I did not retrieve the BBS commentaries in full, so the specifics of the debate rest on abstracts and summaries.

## B1. Public datasets with item-level human judgement plus AU or cue annotation

### Takeaway
The best no-cost resource is Snoek's AU and emotion data: CC BY 4.0, trial-level, with human responses to randomised AU configurations. Importantly, it covers bored, confused, interested and thinking as well as basic emotions. The stimuli are parameters, not videos, so the faces must be re-rendered or requested. CREMA-D and the open RMET and MRMET data are the next most useful.

### Cited Findings
| Dataset | Stimuli | Human item-level data | AU or cue data | Access and licence | Status |
|---|---|---|---|---|---|
| [Snoek AU + emotion data](https://doi.org/10.21942/uva.21261885) | Dynamic GFG stimuli described by parameters; no video files | Per-subject trial TSVs with label; intensity in basic study | Yes, 33 AU amplitudes per trial | CC BY 4.0, 6 MB, open | [V] downloaded |
| [Liu et al. 2022 data](https://doi.org/10.17632/dr853shk56.2) | Reverse-correlation AU data | Category plus valence and arousal ratings | AU-level; 26 of 42 AUs reported as multiplexed | Mendeley Data; licence not checked; modelling code by request | [V] data statement via PMC; contents not inspected |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | 7,442 clips, 91 actors | Per-rater files; per-clip option votes for voice-only, face-only, audiovisual | No AU | ODbL and DbCL; registration form requested; Git LFS, about 7.55 GB | [V] downloaded ratings |
| [RAVDESS](https://zenodo.org/records/1188976) | 7,356 files | Each file rated 10 times; validation data with PLoS paper | Separate [tracking dataset](https://zenodo.org/records/13243600) | CC BY-NC-SA 4.0; free for non-commercial use | [V] record; per-file ratings not inspected |
| [FER+](https://github.com/microsoft/FERPlus) | FER-2013 stills | 10 votes per image across emotion options | No | Labels on GitHub, repo archived; pixels via Kaggle; licence unconfirmed | [S] |
| [RAF-DB](https://openaccess.thecvf.com/content_cvpr_2017/html/Li_Reliable_Crowdsourcing_and_CVPR_2017_paper.html) | About 30,000 images, including compound subset | About 40 labels per image | No | Access terms not verified; third-party listing says non-commercial | [S] |
| [MAFW](https://github.com/mafw-database/MAFW) | 10,045 clips | 11 annotators, 11-dimensional distribution vector | Captions describing brow, eye and mouth cues; landmarks | Free; signed EULA; students need faculty signature; no redistribution | [V] repo page |
| [Aff-Wild2](https://ibug.doc.ic.ac.uk/resources/aff-wild2/) | 564 videos | Consensus frame labels; annotator data not stated | 12 AUs | [EULA](https://sites.google.com/view/dimitrioskollias/databases/aff-wild2); PhD students via supervisor; reply about 14 days | [V] |
| [DISFA](https://service.tib.eu/ldmservice/dataset/disfa) | 27 subjects | No human emotion judgement found | AU intensity | Terms not verified; [listed homepage](https://www.cs.cmu.edu/~jcf/Disfa.html) | [S] |

My profile of the Snoek files:
- Basic emotions, Western: 60 participants, 140,910 trials, 4,659 distinct AU configurations.
- Basic emotions, East Asian: 60 participants, 139,728 trials, 2,323 configurations.
- Conversational, Western: 60 participants, 47,300 trials, 2,318 configurations.
- Conversational, East Asian: 60 participants, 47,460 trials, 2,325 configurations.
- Repeats per configuration are about 15 or 45 (Western basic), 59 to 60 (East Asian basic) and 20 (conversational, mostly).
- Files include `sub_split` and `trial_split` columns, which provide built-in train and test partitions.

Other points:
- Snoek et al. turn hypotheses into predictive AU models for six basic emotions and four conversational signals, test the causal role of AUs, and set a noise ceiling separating AU-explained variance from individual differences. [A] — [Science Advances 2023](https://eprints.gla.ac.uk/289341/). Code: [hypothesis-kernel-analysis](https://github.com/lukassnoek/hypothesis-kernel-analysis) (BSD-3) and [noiseceiling](https://github.com/lukassnoek/noiseceiling). [V]
- The thesis version reports 121,902 trials after removing "other" from a 60-person sample, attributed to Yu et al. 2012. My count from the Western file after the same removal is 119,387. [V] — [thesis chapter](https://lukas-snoek.com/thesis/hypothesis-kernel-analysis.html). The reason for the gap is unknown.
- Liu et al.'s statement says custom code for modelling, experiment and visualisation is available by request from the authors. [V] — [PMC8751635](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8751635/)
- FACSHuman is open-source software on MakeHuman that manipulates almost all FACS AUs and produces images and video. [S] — [Gilbert et al. 2021](https://doi.org/10.3758/s13428-021-01559-9); [GitHub](https://github.com/uni-bremen-agst/FACSHuman). FaReT is GPL-licensed but its database has six emotion poses, with no AU control found. [S] — [Hays et al. 2020](https://doi.org/10.3758/s13428-020-01421-4)

### Inferences
- Snoek's data offers the most direct answer to "what is it about the image". Human AU effects are identified by randomisation, so no mediation assumption is needed on the human side. The matching step is to render the same AU vectors and intervene on single AUs for the model.
- Rendering is the weak link. A FACSHuman avatar is not a GFG face, and amplitude units may not map. Three mitigations: request stimuli or the renderer from Jack's group; check renders with an AU detector; and report results at the level of AU configurations, not pixels. Also check that the VLM behaves sensibly on the avatar domain. Domain shift could masquerade as a strategy difference.
- This data has no human study cost and stays compatible with keeping EU-Emotion on the HPC.

### Gaps
- I did not verify whether Glasgow will share GFG renders. This should be an early email.
- `face_id` values in the files look like render or identity codes that I did not decode.

## B2. Datasets containing complex mental states

### Takeaway
Complex-state data with public human distributions is rare. Snoek's conversational set, the open RMET and MRMET files and your existing EU-Emotion and Mindreading data are the core. MAFW, CEED, ADFES, GEMEP, DAiSEE and EmoReact can extend them, with weaker human or cue data.

### Cited Findings
- Snoek conversational data labels are bored, interested, confused, thinking and other in the Western files. Counts: 11,499, 11,024, 9,880, 8,137 and 6,760. [V, my count]
- MRMET and RMET open data: 60,000-odd TestMyBrain participants reported in secondary coverage. [S] Stimuli sit in a password zip after a short [registration form](https://forms.gle/3miGLTDhNPR2M2v26). [V] — [OSF ahq6n](https://osf.io/ahq6n/). A mailing-list post gives CC-BY-SA 4.0. [S] — [Wilmer post](https://lists.stir.ac.uk/mailman3/hyperkitty/list/face-research-list@lists.stir.ac.uk/thread/TMQ4XS7PIEYNBTEAFDKHTZ2WKQHQUP3I/); the OSF API returned no node licence, so check the licence file. Paper: [Behavior Research Methods 2024](https://link.springer.com/10.3758/s13428-023-02323-x).
- CEED: 243 basic and 237 complex stills, eight actors, at least 50 raters per image in a 4-alternative task. Its complex set leans towards social-sexual expressions. Access route not confirmed. [S] — [PLoS ONE 2020](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6996812/)
- MPI Facial Expression Database: 55 emotional and conversational expressions from 19 German participants, with static and dynamic versions; the 2010 description gives 20 and 58. Terms not found. [S] — [PLoS ONE 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3305299)
- GEMEP: 18 affective states in the corpus; the core set has 145 files, free with a signed form, plus microcoded actions and observer ratings. The official page blocked my fetch. [S] — [GEMEP](https://www.unige.ch/cisa/gemep); [Emotion 2012](https://pubmed.ncbi.nlm.nih.gov/22081890/)
- ADFES: nine emotions including contempt, embarrassment and pride; free for scientific use on request; contact adfes-fmg@uva.nl. [S] — [UvA page](https://aice.uva.nl/research-tools/adfes-stimulus-set)
- DAiSEE: 9,068 snippets, 112 users, labelled for boredom, confusion, engagement and frustration by crowd annotation. Academic use by [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfAkOfiPS3tGy2MFgQAPRMIjRu8L6eBmGiqS-lHqPWSHbJmjA/viewform), whose owner I could not confirm. [S] — [arXiv 1609.01885](https://arxiv.org/abs/1609.01885v6)
- EmoReact: 1,102 children's clips, 17 states including curiosity and uncertainty. Access not found. Consider the ethics of child data. [S] — [record](https://service.tib.eu/ldmservice/dataset/emoreact)
- MAFW: 11 single labels including anxiety, helplessness, disappointment and contempt, plus 32 compound classes. [V]
- Mindreading: one search summary gives 412 mental states, 6 videos each, panel of 10, without a pinned source; you hold this data, so confirm locally. Related system: [el Kaliouby and Robinson](https://www.cl.cam.ac.uk/~pr10/publications/rtv4hci05.pdf) [S]

### Inferences
- A compact test set: EU-Emotion, RMET and CARD, MRMET, and Snoek conversational for strategy claims; use the others only for generalisation checks.
- Bored, confused, thinking and interested overlap with DAiSEE's boredom and confusion, which allows a cross-dataset check of AU-to-state rules in natural video. Labels differ in meaning, so treat it as exploratory.

### Gaps
- No public dataset I found combines complex states, video, option-level human distributions and AU coding on the same natural stimuli. This gap is a possible contribution in its own right.

## B3. Which datasets give a per-item, option-level human distribution

### Takeaway
Yes for Snoek (per trial), open RMET and MRMET (per-person selections), CREMA-D (per clip by modality) and EU-Emotion (your aggregated data). MAFW, FER+ and RAF-DB offer vote-based distributions. Most others only publish hit rates.

### Cited Findings
- The RMET normative workbook has sheets or columns titled "RMET Item-by-item accuracy" and "RMET Item-by-item response selections", plus age, gender, education and ethnicity fields, and it is one of three files: `RMET normative data set.xlsx` (5.3 MB), `MRMET normative data set.xlsx` (6.9 MB), `MRMET and RMET validation data set.xlsx` (5.8 MB). [V, inspected headers for RMET only] — [OSF ahq6n](https://osf.io/ahq6n/)
- CREMA-D `tabulatedVotes.csv` has 22,326 rows, one per clip and modality, with counts for A, D, F, H, N, S. ID prefix 2 is face-only; I confirmed this by matching row values to `summaryTable.csv`. Face-only: median 10 ratings (range 6 to 12), mean majority agreement 0.70, and 31.5% of clips below 0.6. [V, my count] Actor-intended labels are also in the filenames.
- FACES gives downloadable normative ratings from 154 raters for expression and age; registration is case by case. [V] — [FACES](https://faces.mpdl.mpg.de/imeji/)
- KDEF has 4,900 images, registration, all rights reserved; no validation data on its page. [V] — [KDEF](https://www.kdef.se/home/aboutKDEF.html)
- Greenberg et al. state their data are available only on request via a Visitor Agreement with Cambridge. [V] — [PNAS 2023](https://doi.org/10.1073/pnas.2022385119). A later correction exists. [A] — [PNAS 2025](https://doi.org/10.1073/pnas.2526143122)
- Conflict: the RAVDESS Zenodo page says 247 raters rated each file 10 times, while the search summary of the PLoS paper reported other validation pools. [V Zenodo; S paper] — [Zenodo](https://zenodo.org/records/1188976); [PLoS ONE 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5955500)

### Inferences
- With option-level data the primary metric should be a noise-ceiling-normalised divergence between model and human option distributions, with accuracy secondary. This addresses CogGym's concern about means versus distributions, if that claim is confirmed.
- Model distributions need care. Sampling, temperature and log-probability readings give different distributions. Fix one before unblinding.
- Individual-level data in Snoek and the RMET files is what makes the "model fits the mean but not the people" analysis possible. EU-Emotion cannot support it.

### Gaps
- I did not open the MRMET workbooks, so I cannot confirm their option-level columns. Wilmer's post says they carry item-by-item information.
- RAVDESS per-file option distributions were not inspected.

## B4. Existing reverse-correlation or Bubbles data for direct re-use

### Takeaway
Reverse-correlation data is directly reusable at no cost through Snoek and Liu. Public trial-level Bubbles masks for facial expressions were not found. I found no study using a VLM as the observer in reverse correlation, which suggests a real novelty window, subject to a broader search.

### Cited Findings
- Jack et al. 2014: reverse correlation with a generative face model, 60 Western observers. [S] — [Current Biology](https://doi.org/10.1016/j.cub.2013.11.064)
- The Snoek thesis attributes the 60-participant AU data to Yu et al. 2012. [V] — [thesis](https://lukas-snoek.com/thesis/hypothesis-kernel-analysis.html). I did not find a "Dyadic social" dataset under that name.
- Chen et al. 2024 reports cultural dynamic expressions conveying category and intensity. Data statement not retrieved. [Located by title; S for content] — [Current Biology](https://doi.org/10.1016/j.cub.2023.12.001)
- Snoek et al. used individual AU models with MEG to argue for a third pathway. Page blocked, data unverified. [S] — [PNAS](https://www.pnas.org/doi/10.1073/pnas.2423560122)
- Bubbles literature: Blais et al. 2012 on six basic emotions, pain and neutral. [S] — [core.ac.uk](https://core.ac.uk/download/357215329.pdf). Saumure et al. on posed and spontaneous expressions, with 4,032 trials per participant. [S] — [JOV 2018](https://jov.arvojournals.org/article.aspx?articleid=2703424). No open masks were located.
- Human-versus-machine precedent: a Bubbles study verified features with a conventional classifier, not a DNN. [S] — [IEEE 2010](https://ieeexplore.ieee.org/document/5583081/)
- A GAN-plus-reverse-correlation method builds personalised expression prototypes. [S] — [PLoS ONE 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10456187/)

### Inferences
- Two designs follow. **Forward:** present human-tested AU configurations to the model. **Reverse:** run reverse correlation on the VLM with new random AU stimuli, deriving model classification images comparable with human ones. Reverse is cheap at scale and gives a full model AU-to-label mapping.
- Bubbles on EU-Emotion for models alone is possible on the HPC, but comparison needs human Bubbles data, which would require new human collection or data sharing.

### Gaps
- The VLM reverse-correlation null is from two searches. Run arXiv and Google Scholar checks for "classification image" with multimodal models before claiming priority.
- Contacting the Blais and Gosselin groups may yield masks. I found no statement either way.

## B5. Not verified, or excluded by the free-of-charge rule

### Takeaway
Many named datasets could not be verified in this pass. None should be assumed free or usable until its page is checked.

### Cited Findings
- [Wikipedia's list of facial expression databases](https://en.wikipedia.org/wiki/List_of_facial_expression_databases) is a starting point only. [S]
- None of the datasets listed under Gaps below has been checked for licence, so no access claim or URL is made for them.

### Inferences
- Treat BP4D and BP4D+ as out of scope until a no-fee route is confirmed, since you described BP4D as licensed.
- Any set with a commercial licence option (RAVDESS) stays free only under its non-commercial terms. That should cover PhD research, but check for industrial partners.

### Gaps
- Not checked in this pass: DFEW, FERV39k, AffectNet, BP4D and BP4D+, RaFD, NimStim, WSEFEP, MMI, SEWA, AM-FED, AVCaffe, EmoSet, Emotion6, EMOTIC and morph sets. Two searches for BP4D, AffectNet, DISFA and KDEF access returned nothing usable.
- I found no public GoEmotions video variant and did not search for it specifically.

## C1. What claim types get accepted at the target venues, 2022 to 2026

### Takeaway
Papers accepted in main tracks make general claims about methods or measurement, and human data serve as the test bed. Pure "model X differs from humans on task Y" papers fit CogSci, CCN or NeurIPS Evaluations & Datasets (E&D) better.

### Cited Findings
| Paper | Venue | Claim type |
|---|---|---|
| [Fel et al.](https://arxiv.org/abs/2211.04533) [A] | NeurIPS 2022 main | Large-scale trade-off finding plus general training method |
| [Han et al.](https://proceedings.mlr.press/v202/han23d.html) [S] | ICML 2023 | Ground-truth test of a comparison method; limits on it |
| [Muttenthaler et al., Human alignment](https://arxiv.org/html/2211.01201) [S] | ICLR venue not confirmed here | Scale and architecture matter little, dataset and objective matter much |
| [Teneggi and Sulam](https://arxiv.org/abs/2405.19146v2) [S] | NeurIPS 2024 | Concept-importance test with Type I and FDR control, demonstrated with CLIP |
| [Avitan and Golan](https://arxiv.org/abs/2510.23321) [A] | NeurIPS 2025 main | Model recovery showing a standard practice can misidentify |
| [Low-pass filtering](https://arxiv.org/pdf/2602.13859) [S, third-party note lists ICLR 2026] | ICLR 2026 unconfirmed | Simple method plus accuracy-consistency frontier |
| [Geirhos et al.](https://arxiv.org/pdf/2006.16736.pdf) [S] | NeurIPS 2020 | New trial-level metric, error consistency |
| [Klein et al.](https://arxiv.org/abs/2507.06645v2) [S; venue unconfirmed] | Preprint | CIs and a copy model for error consistency; many reported differences not significant |
| [Sucholutsky et al.](https://arxiv.org/pdf/2310.13018) [S] | TMLR 2025 | Unifying framework for representational alignment |
| [Xiang et al.](https://cicl.stanford.edu/publication/xiang2025how/) [S] | CogSci 2025 | Strategy finding via cognitive-model fits |

- NeurIPS 2026 renamed Datasets & Benchmarks to Evaluations & Datasets, treats evaluation as "an object of scientific study in its own right", and says submissions "need not introduce a new model or outperform prior work". Scope includes human-centred evaluation and negative results. [V] — [Call](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets)
- E&D reviewers ask: "Does it change how we measure progress in this area?" and whether it would shift conclusions if applied to existing benchmarks. For methods, "New framing is sufficient". "Negative results are valuable when rigorously supported". Human-centred evaluations require documented ethics approval. [V] — [Reviewing Guidelines](https://neurips.cc/Conferences/2026/EvaluationsDatasetsReviewerGuidelines)
- E&D datasets must be hosted accessibly with Croissant core and RAI metadata, and data and code must be reachable by reviewers without personal request. Non-compliance can justify desk rejection. [V] — [Call](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets)
- The NeurIPS 2026 main call lists neuroscience and cognitive science, and values interdisciplinary work and "in-depth analysis of existing methods". [V] — [Call for Papers](https://neurips.cc/Conferences/2026/CallForPapers)
- Workshops: CogInterp at NeurIPS 2025 focused on how models reach behaviours, not only whether they can. [S] — [site](https://coginterp.github.io/neurips2025). Mechanistic Interpretability ran at ICML 2026. [S] — [listing](https://icml.cc/virtual/2026/workshop/54071). No NeurIPS 2026 listing for either was found.

### Inferences
- EU-Emotion data cannot be redistributed, so an E&D dataset paper is at risk on the reviewer-access rule. Options: publish derived item-level outputs, or use open human data for the core benchmark and report EU-Emotion as a restricted-access validation. Workstream A in `PIVOT.md` should be rechecked against this. Check O'Reilly data rights too.
- Snoek's CC BY licence avoids this problem, which is another reason to centre that dataset.

### Gaps
- Papers at the NeurIPS 2025 D&B track on human psychophysics were not found by search. A manual proceedings pass is needed.
- I found no emotion-focused human-mechanism paper in NeurIPS, ICML or ICLR main tracks. This was search-limited.
- CCN 2026 proceedings were not surveyed, as the search returned nothing relevant.

## C2. What an ICML main-track reviewer needs beyond an empirical case study

### Takeaway
ICML asks for original, rigorous work "of significant interest to the machine learning community", so a face-perception result alone will not carry it. It needs a general method or guarantee, shown on several domains, with faces as the motivating case.

### Cited Findings
- ICML 2026 criteria: claims "must be clearly stated and supported by reproducible experiments and/or sound theoretical analysis", and contributions must be placed in the context of prior work. Application papers should bring "innovative techniques, problems, and datasets that are of interest to the machine learning community". Topics include evaluation methodology, causality and interpretability. [V] — [ICML 2026 Call](https://icml.cc/Conferences/2026/CallForPapers)
- ICML's 2026 reviewer guideline URL returned 404, so I rely on the call. [V]

### Inferences
A reviewer would likely want:
1. A formal estimand that applies beyond faces.
2. At least one theorem or identifiability result, even a simple one.
3. A synthetic ground-truth setting where the correct answer is known, as in Han et al.
4. At least two or three domains, for example AU-emotion (Snoek), objects (ClickMe or THINGS) and mental-state items (MRMET).
5. Comparison with existing metrics, showing conclusions change.
6. Open code and data, with restricted EU-Emotion kept secondary.
7. Sample-size guidance.

Candid estimate: the empirical study alone is a stronger fit for CCN, CogSci or NeurIPS E&D than ICML. This is my judgement, not a statistic.

### Gaps
- No acceptance-rate data specific to human-comparison papers was found.

## C3. Reusable general contributions

### Takeaway
The strongest candidate is a causal feature-alignment framework: it compares randomised human cue effects with interventional model cue effects, adds a noise ceiling and recovery guarantees, and proves that behavioural alignment does not imply it.

### Cited Findings
- Precedents: conditional-independence concept testing with Type I control. [S] — [Teneggi and Sulam](https://arxiv.org/abs/2405.19146v2). Causal abstraction and mediation. [S] — [Geiger et al.](https://arxiv.org/pdf/2301.04709). Noise-ceiling method for classification. [V] — [noiseceiling](https://github.com/lukassnoek/noiseceiling). Error-consistency CIs. [S] — [Klein et al.](https://arxiv.org/abs/2507.06645v2). Metric comparison. [S] — [How aligned are alignment metrics?](https://arxiv.org/html/2407.07530v1). Behavioural RSA method. [S] — [arXiv 2412.00577](https://arxiv.org/pdf/2412.00577)

### Inferences
1. **Causal Feature Alignment (CFA).** Estimand: the vector of average causal effects of cue j on option k, for humans (from randomised stimuli) and for models (from interventions). Score: noise-ceiling-normalised similarity. Candidate theorems:
   - A non-identifiability result: two systems can have equal or perfect error consistency with humans and arbitrary CFA, so behaviour alone cannot lower-bound feature alignment.
   - Finite-sample bias and disattenuation bounds.
   - Trials needed to recover the true model, set against Avitan and Golan.
2. **Prior-evidence decomposition.** A general estimator of how much of a VLM's output is image-independent, with calibration against evidence reliability. It applies to any vision-language task and connects with Deng et al.
3. **Cue-level mediation.** Separate direct effects of a cue from effects passing through a named model component. On the human side only total effects are identified, unless stimuli manipulate the mediator, so this is a model-only analysis and should be labelled as such.
4. **Sequential alignment test.** Adapt betting-based testing to stop early when deciding whether a model is closer to human than a rival, which saves GPU time.
5. **Resource-rational curve fit.** A one-parameter family comparing humans and models, usable on any forced-choice task with option-level data.

Suggested order: CFA as the core, prior-evidence decomposition as the second result, resource-rational fit as the psychiatry application.

### Gaps
- None of these is proved or implemented. The non-identifiability claim looks straightforward by construction but has not been formally written.
- I found no existing metric with exactly this estimand, but "mechanistic alignment" has many near-synonyms, so run a targeted literature check before claiming novelty.

## C4. Upcoming deadlines (as of 2026-11-04)

### Takeaway
CCN 2027 has the earliest confirmed deadlines (Feb 1 and 4, then Apr 1). ICML 2027 dates are unconfirmed but projected for mid to late January, about 11 weeks away. ICLR 2027 has closed. NeurIPS 2027 has no dates yet.

### Cited Findings
| Venue | Conference | Deadline | Status |
|---|---|---|---|
| ICML 2027 | Jul 31 to Aug 6, 2027, Riocentro, Rio de Janeiro; main Aug 2 to 4 | CFP "will be posted soon". Tracker projects abstract Jan 16 and paper Jan 22, 2027 | [Official site](https://icml.cc/) [V] for place and dates; [projection only](https://www.opencurious.com/ai-conference-deadlines/icml-2027) [V, last reviewed Oct 30]. ICML 2026 was Jan 23 and 28 |
| NeurIPS 2027 | Europe; no city or dates | None listed | [Future meetings](https://neurips.cc/Conferences/FutureMeetings) [V]. 2026 was May 4 and 6, notification Sep 24 |
| ICLR 2027 | Main Apr 26 to 28, workshops Apr 29 to 30, 2027; location to be announced | Closed: Sep 18 and 25, 2026 | [Official dates](https://iclr.cc/Conferences/2027/Dates) [V]. Third-party location claims conflict. Suggested workshop deadline Feb 1, 2027 [S] |
| CogSci 2027 | Jul 28 to 31, 2027, Euskalduna, Bilbao | Not posted | [Society page](https://cognitivesciencesociety.org/cogsci-2027/) [V]. Theme "Human Cognition in the Age of AI" from an [aggregator](https://www.beri.net/events/cogsci-2027) [V, checked Oct 3]. CogSci 2026 closed Feb 2 [S] |
| CCN 2027 | Jul 26 to 29, 2027, Edinburgh | Feb 1 abstract and Feb 4 full paper (8-page proceedings); Mar 18 events; Apr 1 two-page abstract | [Official site](https://www.ccneuro.org/) [V] |

### Inferences
- A realistic sequence:
  1. Two-page or 8-page CCN paper on the AU forward test, Feb to Apr 2027.
  2. CogSci 2027, if the deadline stays around early February.
  3. NeurIPS 2027 main or E&D, around May if the pattern holds, for the CFA method.
  4. ICLR 2028, likely autumn 2027, as a fallback.
- ICML 2027 is feasible only if the CFA theorem and multi-domain results are nearly ready, which is unlikely in 11 weeks given renders are not yet validated.
- Plan a Gershman email and a Glasgow email on renders now, since both affect the February scope.

### Gaps
- ICML, NeurIPS and CogSci 2027 deadlines are not official. Recheck their sites.
- Workshop schedules for 2027 are unknown.
