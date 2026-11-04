# Prior work comparing machines with humans on emotion, face and mental-state perception: a novelty map

Search date for everything below: 2026-11-04. Status tags: [PR] = peer-reviewed or accepted at a named venue. [PP] = preprint. "accepted" means the claim comes from the arXiv comment field, not from the proceedings. [abstract] or [snippet] = I read only that much, not the full text.

Bottom line for the study:
- Current Q1 (same confusions?) is already partly answered, for GPT-4o on the MRMET (Strachan et al.).
- "Mechanistic interpretability of VLM emotion" as a bare topic is scooped. An ICML 2026 paper does it on Qwen3-VL.
- What I could not find is any paper linking controlled human psychophysics on faces to causal intervention inside a VLM. No located paper establishes which cues humans use and then tests whether the model's internals causally use the same cues.

## Q1. LLM/VLM emotion recognition and theory-of-mind benchmarks

### Takeaway
Accuracy-level questions are answered and saturating: frontier VLMs beat average humans on the RMET and match them on posed faces. One study went beyond accuracy to compare error structure on the RMET, using GPT-4o, a closed model, with fixed option order. No located study applies option counterbalancing, an image-free baseline, or open-weight internals to the RMET.

### Cited Findings

**RMET and face-based mental-state tests**
- Strachan, Pansardi, Scaliti, Celotto, Saxena, Yi, Manzi, Rufo, Manzi, Graziano, Panzeri, Becchio. "GPT-4o reads the mind in the eyes." arXiv Oct 2024 [PP, full text read]. A research-portal record lists it in Computers in Human Behavior vol. 181, art. 108992, Aug 2026 [PR, venue seen only on a portal page, not on the journal site] — [arXiv](https://arxiv.org/abs/2410.22309); [portal record](https://scholar.hit.edu.cn/en/publications/gpt-4o-reads-the-mind-in-the-eyes/).
  - Model and stimuli: gpt-4o-2024-08-06, RMET and Multiracial RMET, upright and inverted. Humans N = 51 (RMET) and N = 400 (MRMET), against n = 25 and n = 200 model runs per orientation — [paper](https://arxiv.org/pdf/2410.22309).
  - Measured: accuracy, confusion matrices, Spearman similarity of flattened matrices (split into on-diagonal and off-diagonal), error concentration, and "information in errors". Control: human subsamples matched to GPT-4o performance — [paper](https://arxiv.org/pdf/2410.22309).
  - Result: GPT-4o beat humans upright and fell below them inverted. Off-diagonal similarity between human and GPT-4o was weak for inverted faces and significantly lower than for upright faces. GPT-4o errors were far more consistent than human errors. Example: "threatening" was confused with "angry" upright but with "scared" inverted, the least likely error for humans. The authors argue inversion changes human processing quantitatively but GPT-4o's error space qualitatively — [paper](https://arxiv.org/pdf/2410.22309).
  - Causal intervention: none. Stimulus inversion is the only manipulation.
  - Confounds the authors leave open: alternatives kept in the fixed published order, labelled (a) to (d), temperature 0.8 and top_p 0.9. The authors flag that faces and response options may be in the training data, potentially inflating performance — [paper](https://arxiv.org/pdf/2410.22309).
- Refoua, Elyoseph, Piterman, Geller, Meinlschmidt, Hadar Shoval. Scientific Reports, Feb 2026 [PR, abstract]. ChatGPT-4, ChatGPT-4o and Claude 3 Opus, tested twice on White, Black and Korean RMET versions, against normative data. GPT-4o scored 83.3%, 94.4% and 86.1% (85th, 94th and 90th percentile). Claude 3 Opus was near chance. No error or causal analysis appears in the abstract — [Nature](https://www.nature.com/articles/s41598-026-39292-y); [PubMed](https://pubmed.ncbi.nlm.nih.gov/41720888/).
- Akben, Gude, Ajjan. "Silicon Minds versus Human Hearts." arXiv Aug 2025 [PP, abstract]. MLLMs outperform individual humans on RMET and MRMET, but aggregated human groups beat aggregated models — [arXiv](https://arxiv.org/abs/2508.08830). Follow-up on GPT-5 mini: Scientific Reports, Mar 2026 [PR, abstract], with the same reversal under plurality voting — [DOI](https://doi.org/10.1038/s41598-026-45331-5).
- Piterman, Dery, Bar, Meinlschmidt, Geller, Shoval, Elyoseph, Refoua. Scientific Reports, Sep 2026 [PR, abstract]. Three Gemini models on EU-Emotion bodily-gesture and vocal stimuli (basic emotions), compared to published human benchmarks. Near-human on voice, below human on gesture, with a positivity bias. This is the only located use of EU-Emotion with generative AI. It does not cover the face channel or mental states — [DOI](https://doi.org/10.1038/s41598-026-58013-z). A web-search summary claimed the models were not shown the stimuli. The abstract does not say that, and I did not verify it in the full text.

**Facial expression and emotion benchmarks**
- Nelson, Winbush, Siddals, Flathers, Allen, Torous. npj Digital Medicine, Oct 2025 [PR, abstract]. GPT-4o, Gemini 2.0 Experimental and Claude 3.5 Sonnet on NimStim. GPT and Gemini matched or exceeded humans. Fear was often misclassified — [Nature](https://www.nature.com/articles/s41746-025-01985-5).
- Lu et al. "GPT as Psychologist?" CVPR Workshops 2024 [PR, snippet]. GPT-4V is strong on AU detection and weaker on expression recognition — [CVF](https://openaccess.thecvf.com/content/CVPR2024W/CVPM/papers/Lu_GPT_as_Psychologist___Preliminary_Evaluations_for_GPT-4V_on_CVPRW_2024_paper.pdf).
- Zhang, Li, Qian et al. FERBench. arXiv Nov 2025 [PP, abstract]. 20 MLLMs on four FER datasets posed as VQA. Reasonable classification, weak reasoning and interpretability — [arXiv](https://arxiv.org/abs/2511.00389).
- Schuhmann, Kaczmarczyk, Rabby et al. EmoNet-Face. NeurIPS 2025 Datasets and Benchmarks [PR, snippet]. 40 emotion categories on synthetic faces with expert annotation; a custom model reaches expert-level agreement — [NeurIPS](https://proceedings.nips.cc/paper_files/paper/2025/hash/8b2dd02001e72f65091115fb7f39d8c7-Abstract-Datasets_and_Benchmarks_Track.html).
- Hu, You, Xu et al. EmoBench-M. arXiv Feb 2025 [PP, abstract] — [arXiv](https://arxiv.org/abs/2502.04424). Sabour et al. EmoBench. ACL 2024 [PR, snippet]; 400 hand-crafted text items, GPT-4 below average human — [arXiv](https://arxiv.org/abs/2402.12071).
- MER and DFEW engineering line: Emotion-Qwen (DFEW WAR 78.31) — [arXiv](https://arxiv.org/html/2505.06685v3); MLLM-for-MER survey — [arXiv](https://arxiv.org/pdf/2509.24322); MER-OV challenge entry — [arXiv](https://arxiv.org/abs/2408.11286). These optimise accuracy. None makes a human comparison of strategy.
- AU-grounded engineering: TAG, arXiv Feb 2026 [PP, abstract], constrains reasoning to AU regions to improve FER — [arXiv](https://arxiv.org/abs/2602.18763). These add AUs as training supervision. They do not ask whether an unmodified model uses AUs.

**Structured comparison with human ratings**
- Asanuma, Koide-Majima, Nakamura, Horii, Nishimoto, Oizumi. arXiv May 2025 [PP, full text]. Gemini-2.0-flash on both datasets, GPT-4.1 and Molmo-7B-D on Cowen and Keltner only (550 and 2,184 clips, 80 and 34 categories). Uses RSA and Gromov-Wasserstein optimal transport. RDM r about 0.56 for Gemini, 0.49 for GPT-4.1, 0.23 for Molmo. Item-level matching is far weaker than category-level matching. Models fail where social or narrative context drives the emotion. Correlational; no causal test. Notably, this is a Molmo human-comparison datapoint on evoked emotion — [arXiv](https://arxiv.org/html/2505.12746).
- Santavirta, Wu, Suominen, Nummenmaa. Imaging Neuroscience, Sep 2025 [PR, abstract]. GPT-4V annotated 138 social features for 468 images and 234 videos. Humans N = 2,254, with fMRI N = 97. Human-like feature structure, and similar brain maps from either annotation source — [DOI](https://doi.org/10.1162/IMAG.a.134).
- Evaluating LLM capacity to interpret emotions in images. PLOS ONE, Jun 2025 [PR, snippet]. GPT-4 correlations of 0.87 (valence) and 0.72 (arousal) — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12133009/).

**Theory of mind**
- Kosinski. PNAS 2024 [PR, snippet] — [PNAS](https://www.pnas.org/doi/10.1073/pnas.2405460121). Ullman. arXiv 2023 [PP, snippet]; trivial alterations overturn apparent ToM success — [arXiv](https://arxiv.org/abs/2302.08399). Strachan et al. Nature Human Behaviour 2024 [PR, snippet]; GPT-4 at or above human on most ToM tests, weak on faux pas — [Nature](https://www.nature.com/articles/s41562-024-01882-z). Ullman's critique is followed up by SCALPEL — [arXiv](https://arxiv.org/html/2406.14737v1).
- "Re-evaluating Theory of Mind evaluation in large language models." Phil Trans R Soc B 2025 [PR, snippet]. Argues the disagreement stems from unclarity over whether models should match human behaviour or the underlying computation — the supervisor's point, made for text ToM — [journal](https://royalsocietypublishing.org/rstb/article/380/1932/20230499/235070/Re-evaluating-Theory-of-Mind-evaluation-in-large).
- Zhang, Sun, Yi, Yang. arXiv Jul 2026 [PP, abstract]. Nine frontier VLMs on the Keysar Director Task and Frith-Happé triangles, against child, adult, HF-ASD and TD groups. 78% egocentric error without CoT. Profile more than 3x closer to HF-ASD than TD on triangles. Group-level; no internal analysis — [arXiv](https://arxiv.org/abs/2608.00261).
- Zhang, Feng, Wang et al. arXiv Jun 2025 [PP, full text]. 111 VLMs on 900 controlled gaze-target photos, 65 humans (about 91% correct). Most VLMs at chance. Uses Geirhos-style error consistency. Top VLMs lack the human View effect. The head-versus-eye cue hypothesis is left explicitly untested — [arXiv](https://arxiv.org/html/2506.05412v1).
- Multimodal ToM benchmarks: MoMentS (humans 91.0% vs best model 67.6%) — [listing](https://lacuna.tiptreesystems.com/work/moments-a-comprehensive-multimodal-benchmark-for-theory-of-mind/wrk_dcc67218c967ecd5b5661acdcba68d8f); MuMA-ToM, AAAI [PR] — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/32142/34297); MMToM-QA — [listing](https://www.alphaxiv.org/abs/2401.08743); Video ToM on Social-IQ 2.0 — [arXiv](https://arxiv.org/pdf/2406.13763).

### Inferences
- Accuracy comparison on RMET is not a publishable question for NeurIPS/ICML. It has been done at least three times, including at GPT-5 mini level.
- Strachan's fixed option order may contribute to GPT-4o's high error consistency. Fixed text gives a model the same token context on every run, while humans differ from each other. This is my reading, not a tested result. It would be a direct and cheap replication target.
- Item-level matching being far weaker than category-level matching (Asanuma) and object-level vs image-level (Rajalingham, below) suggests aggregate RSA overstates similarity. Item-level measures are the informative level.

### Gaps
- "Do VLMs see emotions?" and a Dominguez-Catena GPT-4 vs human FER paper could not be located under those names. Dominguez-Catena's located work is FER dataset bias without GPT-4 — [arXiv](https://arxiv.org/pdf/2210.05332). The nearest is Hosseini et al., GPT-4o-mini fairness — [arXiv](https://arxiv.org/pdf/2502.11049).
- ToMBench authorship and venue were not verified. Only a secondary description was located.
- RMET with open-weight VLMs (Qwen, Gemma, Molmo): no paper located.
- EU-Emotion face-channel use with any LLM/VLM: no paper located.

## Q2. DNNs and face or emotion perception

### Takeaway
For CNNs, inversion, confusion and category-selectivity effects are well studied. Reverse correlation has been run on DNNs for identity, but I found no equivalent for expression. Nearly all of this is CNN-based. None of it touches VLMs. Transformers have barely been tested on these signatures.

### Cited Findings
- Kragel, Reddan, LaBar, Wager. Science Advances 2019 [PR, snippet]. EmoNet (AlexNet-derived), about 25,000 images, 20 categories, plus fMRI N = 18. Emotion-category codes in the visual system. Evoked emotion from scenes, not facial expression — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6656543).
- Zhou, Yang, Meng, Zhou. Science Advances 2022 [PR, abstract]. Expression-selective units in identity-trained VGG-Face show hallmarks of human expression recognition, namely facial expression confusion and categorical perception. Object-trained and untrained networks have similar units without those human-like hallmarks. Web-search summaries of this paper misreport it; the abstract is the authority — [DOI](https://doi.org/10.1126/sciadv.abj4383).
- Dobs, Yuan, Martinez, Kanwisher. PNAS 2023 [PR, snippet]. Face inversion and other human signatures appear in CNNs trained on face recognition, not object or face detection tasks, with face experience matched — [DOI](https://doi.org/10.1073/pnas.2220642120). Dobs et al. Science Advances 2022 [PR, title verified]: functional specialisation emerges spontaneously — [DOI](https://doi.org/10.1126/sciadv.abl8913).
- Tian et al. Frontiers in Computational Neuroscience 2022 [PR, snippet]. Stronger FIE in VGG-Face than in object DCNN — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9124772/).
- Yovel, Grosbard, Abudarham. Proc R Soc B 2023 [PR, abstract]. Larger inversion effect for faces than objects; face-like inversion for birds only in a specialised network — [DOI](https://doi.org/10.1098/rspb.2023.0093).
- Daube et al. Patterns 2021 [PR, snippet]. Humans rated generated 3D faces for similarity to four identities; ratings were predicted from five DNNs. Uses redundancy analysis, reverse correlation and generalisation gradients. Argues behavioural match is insufficient without equating functional features. Identity only — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8515012).
- Schyns, Snoek, Daube. Trends in Cognitive Sciences 2022 [PR, snippet]. Three degrees of equivalence: same responses; same stimulus features; same computations on those features. Proposes generative psychophysics applied to DNNs — [Glasgow](https://eprints.gla.ac.uk/278852); [PubMed](https://www.pubmed.gov/36216674/).
- Snoek et al. Science Advances 2023 [PR, snippet]. Seven AU-based emotion models tested with hypothesis kernel analysis. Models fit Western better than East Asian observers. Humans and AU models only, no DNN or LLM — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9916981/).
- Jack, Garrod, Yu, Caldara, Schyns. PNAS 2012 [PR, snippet]. Reverse-correlated mental representations of dynamic expressions — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3358835). Delis et al. J Vision 2016 [PR, snippet]; low-dimensional AU structure of categorisation — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4927208).
- Jozwik, O'Keeffe, Storrs et al. PNAS 2022 [PR, snippet]. Euclidean BFM distance explains human face dissimilarity as well as DNNs — [PDF](https://kriegeskortelab.zuckermaninstitute.columbia.edu/sites/default/files/content/JozwikOKeeffeStorrs_etal_2022_PNAS.pdf).
- Abudarham line: critical features, and DCNN sensitivity to them when face-trained — [2024 preprint PDF](https://people.socsci.tau.ac.il/mu/galityovel/files/2024/11/2024.06.08.597949v1.full11.pdf).
- Baek, Song, Jang, Kim, Paik. Nature Communications 2021 [PR, snippet]. Face-selective units in untrained networks — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8677765/). Vinken, Prince, Konkle, Livingstone. Science Advances 2023 [PR, title verified] — [DOI](https://doi.org/10.1126/sciadv.adg1736).
- Baker and Elder. iScience 2022 [PR, title verified]: deep models fail to capture configural shape perception — [DOI](https://doi.org/10.1016/j.isci.2022.104913).
- Reviews: Phillips and White, British Journal of Psychology 2025 [PR, abstract] — [DOI](https://doi.org/10.1111/bjop.12794); O'Toole and Castillo, Annual Review of Vision Science 2021 [PR, title verified] — [DOI](https://doi.org/10.1146/annurev-vision-093019-111701).
- FER DNN vs human attention (CNN-level precedent for the "same information" question):
  - Park and Wallraven. arXiv 2021 [PP, abstract]: click-uncovered human attention vs CAM, GradCAM, extremal perturbation in 2AFC FER — [arXiv](https://arxiv.org/abs/2110.04481).
  - Gaya-Morey et al. Multimedia Tools and Applications 2024 [PR, abstract]: 12 networks, global heatmaps compared with facial regions — [arXiv](https://arxiv.org/abs/2401.11835).
  - Li and Ying. Frontiers in Computational Neuroscience 2022 [PR, abstract]: DCNN occlusion on emotion categorisation — [DOI](https://doi.org/10.3389/fncom.2022.1054421).
- Borji and Lin. "White Noise Analysis of Neural Networks." arXiv 2019 [PP, abstract; arXiv lists no venue, and I recall ICLR 2020 without verifying it]. Classification images applied to deep networks; object-level, not faces — [arXiv](https://arxiv.org/abs/1912.12106).

### Inferences
- Zhou 2022 means "confusion structure similar to humans" arose in a CNN from face experience alone, four years ago. Q1 alone therefore has a precedent outside VLMs too.
- Schyns's three degrees give a direct formalisation of the supervisor's "HOW" question. The VLM literature sits almost entirely at Degree 1. Daube reached Degree 2 for identity in CNNs. Degree 2 for expression or mental states in VLMs appears unoccupied.

### Gaps
- Mistry, and "Khaligh-Razavi" in a face or emotion context, could not be located from the names given. Please check the exact references.
- O'Toole's specific expression-related DNN work was not reviewed.
- Availability of Glasgow generative face stimuli and of human trial data for reuse was not verified, and it affects feasibility.

## Q3. General methodology for human-vs-model comparison

### Takeaway
This literature supplies the standards a reviewer will apply. Accuracy and aggregate confusion are not enough. Controlled manipulation and image-level or trial-level analysis are expected. Every principle below is established; few have been applied to emotion or face tasks in VLMs.

### Cited Findings
- Geirhos et al. ICLR 2019 [PR, snippet]: texture bias in ImageNet CNNs, shifted by training data — [arXiv](https://arxiv.org/abs/1811.12231).
- Geirhos, Meding, Wichmann. NeurIPS 2020 [PR, snippet]: error consistency. CNN-human agreement is only slightly above chance — [arXiv](https://arxiv.org/abs/2006.16736).
- Rajalingham et al. J Neuroscience 2018 [PR, snippet]: more than 1 million trials. Models match primates at object level but not image level — [J Neurosci](https://www.jneurosci.org/content/38/33/7255). This matters for Q1: pair-level match is weak evidence.
- Brain-Score. bioRxiv 2018 [PP, snippet] — [bioRxiv](https://www.biorxiv.org/content/10.1101/407007).
- Feather, Leclerc, Madry, McDermott. Nature Neuroscience 2023 [PR, snippet]: model metamers from late stages are often unrecognisable to humans; models have idiosyncratic invariances — [PDF](https://mcdermottlab.mit.edu/papers/Feather_etal_2023_deep_metamers.pdf).
- Bowers et al. BBS 2023 [PR, snippet]: DNNs account for almost no results from psychology experiments; models should explain experiments that manipulate independent variables rather than compete on prediction — [Cambridge](https://cambridge.org/core/services/aop-cambridge-core/content/view/ABCE483EE95E80315058BB262DCA26A9/S0140525X22002813a.pdf/deep-problems-with-neural-network-models-of-human-vision.pdf).
- Biscione et al. MindSet: Vision. arXiv 2024 [PP, abstract]: 30 psychological findings as manipulated datasets, including Thatcherised faces — [arXiv](https://arxiv.org/abs/2404.05290).
- Mehrer et al. PNAS 2021 [PR, snippet] — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7923360/). Peters and Kriegeskorte. Nature Human Behaviour 2021 [PR, snippet] — [arXiv](https://arxiv.org/abs/2109.03351v1).
- Muttenthaler et al. ICLR 2023 [PR, snippet]: linear transforms raise alignment with THINGS odd-one-out; architecture matters little, training data and objective matter more — [arXiv](https://arxiv.org/html/2211.01201v5).
- Baker, Lu, Erlikhman, Kellman. PLOS Computational Biology 2018 [PR, title verified] — [DOI](https://doi.org/10.1371/journal.pcbi.1006613).
- Fel, Felipe, Linsley, Serre. NeurIPS 2022 [PR, snippet]: ClickMe feature maps for 84 DNNs; harmonisation aligns model and human visual strategy — [arXiv](https://www.arxiv.org/pdf/2211.04533).

### Inferences
- Bowers's critique applies directly. RMET benchmarks are observational. A factorial, manipulated-stimulus design would fit the demand from both Bowers and Schyns.
- Rajalingham and Muttenthaler together warn that a human-model confusion correlation may be inflated by shared category statistics, or be recoverable after a linear transform. Neither is evidence of shared mechanism.

### Gaps
- Fel-style harmonisation, ClickMe or metamers applied to VLMs on faces: nothing located.

## Q4. Cognitive modelling and psychophysics on LLMs and VLMs

### Takeaway
"Machine psychology" is an established field, with a Science Advances synthesis in Oct 2026. VLM psychophysics grew quickly in 2025-2026, mostly on low-level and mid-level vision. Depth is the closest methodological analogue to what the study needs; faces and emotion are not covered. That leaves a real gap, but the template is public and easy to copy.

### Cited Findings

**Text LLMs**
- Binz and Schulz. PNAS 2023 [PR, snippet] — [arXiv](https://arxiv.org/abs/2206.14576). Aher, Arriaga, Kalai. ICML 2023 [PR, snippet] — [PMLR](https://proceedings.mlr.press/v202/aher23a.html). Hagendorff, Fabi, Kosinski. Nature Computational Science 2023 [PR, snippet] — [arXiv](https://arxiv.org/abs/2306.07622). Lampinen, Dasgupta et al. PNAS Nexus 2024 [PR, title verified] — [DOI](https://doi.org/10.1093/pnasnexus/pgae233). Binz et al. Centaur, Nature 2025 [PR, title verified] — [DOI](https://doi.org/10.1038/s41586-025-09215-4).
- Hagendorff, Dasgupta, Binz, Chan, Lampinen, Wang, Akata, Schulz. "Machine psychology for language models." Science Advances, 28 Oct 2026 [PR, abstract]. Argues for computational insight beyond performance — [DOI](https://doi.org/10.1126/sciadv.aec1912). It appeared a week before this search; check whether it discusses VLMs.
- Peterson, Abbott, Griffiths 2017 [PR, snippet] — [IJCAI](https://www.ijcai.org/proceedings/2017/697). Marjieh et al. predicting similarity judgments from LLMs — [listing](https://lacuna.tiptreesystems.com/work/predicting-human-similarity-judgments-using-large-language-models/wrk_278551bb7ee3f1d797f36241a43cd69d). Marjieh, Sucholutsky, van Rijn, Jacoby, Griffiths. "Large language models predict human sensory judgments across six modalities." arXiv 2023 [PP, title verified] — [arXiv](https://arxiv.org/abs/2302.01308). This warns that text models can predict perceptual similarity with no vision at all.
- PNAS paper on LLM-derived conceptual representations (LLaMA3-70B) — [PNAS](https://www.pnas.org/doi/10.1073/pnas.2512514122).

**VLM psychophysics and illusions**
- Liu, Kotseruba, Tsotsos. Odd-One-Out Depth. ECCV 2026 accepted [PR, abstract]. Controls individual pictorial depth cues, varies referring expressions, and defines a vision-vs-language sensitivity metric. 37K images, 12 open-source VLMs. This is the closest template for cue-level plus vision-vs-language separation; it uses depth, not faces — [arXiv](https://arxiv.org/abs/2607.01503).
- Zhang, Golovanevsky, Domini, Tompkin. ECCV 2026 accepted [PR, abstract]. VLMs output anchors, not graded slant; a failure at the representation-to-language interface — [arXiv](https://arxiv.org/abs/2606.06714).
- Wen, Madinei, Eckstein. NeurIPS 2026 accepted [PR, abstract]. Counterfactual Semantic Saliency: causal object ablation; 16,603 human responses on 307 scenes. VLMs over-rely on size, centrality and saliency and are less sensitive to people. Black-box; scenes, not faces — [arXiv](https://arxiv.org/abs/2605.13047).
- Hernandez-Camara et al. Neural Networks 2026 [PR, snippet] — [arXiv](https://arxiv.org/abs/2508.10367).
- Matta et al. arrow-of-time, arXiv Oct 2025 [PP, abstract] — [arXiv](https://arxiv.org/abs/2510.26241). Wick, visual search with thinking tokens as an RT analogue, arXiv Jun 2026 [PP, abstract] — [arXiv](https://arxiv.org/abs/2606.25066). Hu, DuTell, Girshick, Corbett, boundaries on ambiguous images, NeurIPS 2026 BiAlign Workshop accepted [PR, abstract] — [arXiv](https://arxiv.org/abs/2603.24730).
- Illusions: IllusionVQA — [arXiv](https://arxiv.org/html/2403.15952v1); Ullman illusion-illusion — [arXiv](https://arxiv.org/abs/2412.18613); Yang et al. survey — [arXiv](https://arxiv.org/html/2508.12422v1), which does not cover faces or pareidolia.
- "Visual language models show widespread visual deficits on neuropsychological tests." Nature Machine Intelligence 2026 [PR, snippet] — [Nature](https://www.nature.com/articles/s42256-026-01179-y).
- Chen, Soydaner, Saunders. Pareidolia. arXiv Mar 2026 [PP, abstract]. CLIP, LLaVA-1.5, ViT, YOLOv8, RetinaFace. LLaVA over-calls Human on 73% of non-human pareidolic images, especially negative emotion — [arXiv](https://arxiv.org/abs/2603.03989).

**Social and face impressions**
- Wolfe, Dangol, Hiniker, Howe. AIES 2024 [PR, abstract]. 43 CLIP models learn human-like facial impressions, with societal consistency predicting reflection — [arXiv](https://arxiv.org/abs/2408.01959).
- Kang et al. arXiv May 2026 [PP, abstract]: MM-OCEAN separates "perception" from "prejudice" in personality with grounding metrics — [arXiv](https://arxiv.org/abs/2605.22109).
- Z. Li, Hung, Raman. GesBench. arXiv Oct 2026 [PP, full text]. Details under Q6(a) — [arXiv](https://arxiv.org/abs/2610.15129).

**Emotion concepts in text models**
- Sofroniew et al., Anthropic, Apr 2026: 171 emotion vectors with circumplex geometry in Claude Sonnet 4.5 [secondary sources only] — critique by Peiris [PP], which argues the method cannot separate emotions from correlated situations — [arXiv](https://arxiv.org/pdf/2604.13466). Open-weight replication in Gemma-2-27B: valence r = 0.72, arousal not replicated [PP] — [arXiv](https://arxiv.org/pdf/2609.22208).
- Wu, Wang, Yan, Tang. Computers in Human Behavior 2026 vol. 183 [PR, snippet]: human-aligned emotion geometry across Gemma and Llama, using SAEs — [PolyU](https://www.polyu.edu.hk/lst/research/publications/journal-papers/2026/0513-decoding-and-controlling-emotion/).
- C. Wang et al. Emotion circuits, arXiv Oct 2025 [PP, abstract]: text only, ablation and enhancement — [arXiv](https://arxiv.org/abs/2510.11328).

### Inferences
- The Peiris critique carries over to faces: an image-derived "emotion vector" may encode posers, dataset or situation rather than emotion.
- Marjieh's sensory-judgment result is the strongest argument for an image-free baseline. Text alone can look like perception.

### Gaps
- Gershman on LLMs, ToM or neural-network models of cognition: three differently worded searches found nothing authored by him. This reflects my search coverage, not evidence that no such writing exists. His site or Scholar page should be checked directly.
- Griffiths-lab work comparing VLM representations with human psychological embeddings on faces: none located.

## Q5. MCQ order bias and prompt confounds, for the study's correct-first 6AFC

### Takeaway
This confound is severe and well documented. The study's correct-answer-first design is the exact control-group condition that literature uses to show inflation. The option-determined-by-target design is a further problem: answer-identifier and option-text information contaminate activation contrasts. A 2026 preprint shows steering effects can follow answer identifiers rather than judgments.

### Cited Findings
- Zheng, Zhou, Meng, Zhou, Huang. ICLR 2024 Spotlight [PR, abstract]. 20 LLMs. Selection bias stems mainly from token bias on option IDs — [arXiv](https://arxiv.org/abs/2309.03882).
- Zhu et al. arXiv 2025 [PP, snippet]. On GPQA Diamond, the group with the correct answer always first consistently outperformed all other groups — [arXiv](https://arxiv.org/pdf/2506.04734).
- Wiegreffe, Tafjord, Belinkov, Hajishirzi, Sabharwal. ICLR 2025 Spotlight [PR, snippet]. Mechanistic analysis of MCQ: answer-symbol choice is causally attributed to one middle layer's self-attention; sparse heads with distinct roles — [arXiv](https://arxiv.org/abs/2407.15018). This matters because an "emotion" effect at a middle layer in an MCQ prompt may be answer machinery.
- Gao et al. "What Does Activation Steering Control?" arXiv Aug 2026 [PP, full text to char 100,000]. Cross-Encoding Steering Evaluation, in text models. Effects followed the extraction index more than the semantic label across all five mappings on NormBank (Qwen favoured semantic label). Extracting the direction at scenario end rather than pre-answer cut the effect by 98.3%. A low-rank subspace with 15.4% of squared norm retained 96.3% of the effect. MCQ and open-ended results diverged. Recommends frozen remapping, factorial identifier and row controls, and separate open-ended evaluation — [arXiv](https://arxiv.org/html/2608.22985).
- Li, Qiu, Xu. arXiv Sep 2026 [PP, abstract]. VLM result: under CoT-prefix scoring, Qwen2.5-VL-7B falls from 80.76% to 45.48%; 93.54% of predictions select the first slot across permutations — [arXiv](https://arxiv.org/abs/2609.29278).
- Zhao, Han, Hovy. arXiv Jun 2026 [PP, abstract]. Multimodal A/B forced choice in both option orders, hidden states taken at the pre-answer state. A VLM precedent for order-crossed internal analysis — [arXiv](https://arxiv.org/abs/2606.08394).
- Kumaran. arXiv Jun 2026 [PP, snippet]: confidence vectors built from trials balanced across option positions, to avoid option-specific representations — [arXiv](https://arxiv.org/pdf/2606.29490).
- Ye et al. EMNLP 2026 Findings accepted [PR, abstract]: source context strongly changes steering success; signals come from execution-boundary states — [arXiv](https://arxiv.org/abs/2607.25270).
- Sandbagging study: label priors and displayed position are tied together by design and cannot be separated — [arXiv](https://arxiv.org/pdf/2604.26206). Removing letter identifiers still gives a first-position shift in question generation — [arXiv](https://arxiv.org/html/2605.01846v1).
- Human forced-choice precedent: Russell, Psychological Bulletin 1994 [PR, title verified] — [DOI](https://doi.org/10.1037/0033-2909.115.1.102). I recall its critique of forced-choice emotion labelling but did not re-read it.
- Good-practice contrast: Zhao et al. (ESN paper) used randomised option order — [arXiv](https://arxiv.org/html/2608.17102). CMES avoids option lists by scoring next-token label words — [arXiv](https://arxiv.org/html/2609.34742).

### Inferences
Everything in this list is my reasoning about the study's own design, not a published finding.
- Correct-first on all 243 items means accuracy cannot be interpreted. It mixes perception with a first-position prior.
- When the option set is fully determined by the target label, target and option text are perfectly confounded. Any decoder, probe or contrast computed across items can read the target from the prompt alone, without any image. Any pre-answer activation contrast between mental states is therefore compromised until shown otherwise.
- The same design prevents testing Q2. A confusion cell exists only if the two labels co-occur, so "pair-specific vs generic difficulty" is partly fixed by which distractors were assigned. Crossing option sets with images is required.
- Existing results built on these prompts should be treated as pilot data. Minimum repair: permute position; cross image with option set; run image-free, scrambled-image and label-only baselines; run a free-text or next-token readout (the CMES style) as a second measure.

### Gaps
- I found no paper on option-order or label-order effects specifically in VLM emotion recognition, nor an image-free baseline on RMET or FER. Query n12 returned only emotion-cause-extraction work. Query x3 returned one irrelevant hit.
- Wang et al. and "Option order" as named references were not pinned down. Nearest located: Zheng above, and the Megagon order-sensitivity paper — [Megagon](https://megagon.ai/order-matters-llm-sensitiv-mult-choice-tsks/).

## Q6. Scoop audit for candidate ideas (a) to (f), and the search log

### Takeaway
Idea (b) as a bare topic is scooped. Ideas (c), (d) and (e) are partly occupied, with emotion-specific human-linked versions still free. Ideas (a) and (f) on VLMs are open except for Strachan's inversion condition.

### Cited Findings

**(a) AU manipulation, reverse correlation, Bubbles on VLMs: NOT SCOOPED, nearest work below**
- No reverse-correlation, Bubbles or classification-image study on a VLM or LLM with face stimuli was located.
- Nearest:
  - GesBench [PP], arXiv Oct 2026. Seven audio-video models, MELD, MIntRec2, Seamless clips. Masks regions around human gaze from eye tracking. Qwen2.5-Omni is sensitive; Gemini and Ming-Lite are nearly flat. No face-region analysis. The authors state it measures sensitivity, not causal importance — [arXiv](https://arxiv.org/html/2610.15129).
  - Kataria, Ghosh, Ye, Dhall. VideoFace-X-LLM [PP], arXiv Oct 2026. Blurs top attention patches and peak frames on a fine-tuned InternVideo2.5. Causal scores 1.44 to 1.93 spatial. No human comparison; AUs are evaluated separately, not used as the explanation — [arXiv](https://arxiv.org/html/2610.17791).
  - Wen et al. (object ablation), Liu et al. (cue control). Both listed under Q4.

**(b) Mechanistic interpretability of VLM emotion or social perception: SCOOPED AS A TOPIC. Read these first.**
- C. Zhang, Sun, Xie, Tian. ICML 2026 accepted [PR, full text to char 100,000]. Qwen3-VL-4B main, with 8B and LLaVA-OneVision-1.5. Emotional vs neutralised image pairs (250 pairs; edited via Nano Banana). Steering-vector attribution and activation patching. Reports an Adapt-Aggregate-Execute stage model with critical layer l19. Patching gains of +15.0%, +16.2%, +17.3%. Head knockout reduces emotional alignment. VEENA improves MER-UniBench from 58.1% to 64.8%. No human behavioural comparison, no AU analysis, no confusion analysis. Same model family and target venue as the study — [arXiv](https://arxiv.org/html/2605.21980).
- Zhao, Sun, Schuller, Sisman. arXiv Aug 2026 [PP, full text]. Gemma-4-12B-it, MiniCPM-o-4.5, Qwen2.5-Omni-7B. AffectNet and MSP-Podcast, five emotions, randomised MCQ. Emotion-sensitive neurons via contrastive activation; deactivation drops Gemma-4 matched-emotion FER from 60.47 to 34.67, steering adds +16.46; cross-modal transfer. No human data, no confusion matrices. Authors note late-layer ESNs may reflect answer selection — [arXiv](https://arxiv.org/html/2608.17102).
- Xing, Liu, Yuan, Li, Gao, Zhao, Fu, Kalviainen. CMES. arXiv Sep 2026 [PP, read to char 200,000]. Qwen3-VL-8B, InternVL3.5-8B, Ministral-3-8B, with Gemma-3 held out. Text and images including AffectNet and RAF-DB, six Ekman emotions. Image-derived vectors share low-dimensional geometry. Steering +4.96 log-probability on AffectNet and changes prediction on about 47% of RAF-DB images. Maps cross architectures. Human references are valence-arousal norms only. Patch maps highlight brow, mouth and eyes, but the authors state these are qualitative and not causal. Mentions anger-disgust and fear-surprise neighbours but does not compare them to human matrices systematically — [arXiv](https://arxiv.org/html/2609.34742).
- Others: Sun et al. EmoMM with CHASE [PP], video marginalised in MER — [arXiv](https://arxiv.org/abs/2605.01024). Zhou, FearCaut-Qwen [PP], affective steering with SDT — [arXiv](https://arxiv.org/abs/2610.11986). Golovanevsky et al. NOTICE, NAACL 2025 [PR, snippet], includes FER — [arXiv](https://web3.arxiv.org/abs/2406.16320).
- Neighbouring ToM mechanistics, all text: CoSToM, ACL 2026 accepted [PR, abstract] — [arXiv](https://arxiv.org/abs/2604.10031); Steele et al. belief-reality routing [PP] — [arXiv](https://arxiv.org/abs/2607.11945).

**(c) Perceptual evidence vs conceptual label semantics: PARTLY. General yes, emotion no.**
- Barua, Mahowald, Rudman. arXiv Oct 2026 [PP, full text to char 100,000]. Four open VLMs, 2,765 conflict items, blur and noise crossed 6x6. Fits sJ = betaT*eT + betaV*eV + pi0 with R2 about 0.69 to 0.86. Cleanly separates evidence strength from integration bias. Behavioural, no humans, no emotion. The most reusable template — [arXiv](https://arxiv.org/html/2610.19407).
- Fang et al. arXiv Sep 2026 [PP, full text]. Four-stage diagnostics on word completion; statistical, no intervention, no emotion — [arXiv](https://arxiv.org/html/2609.35272).
- "Vision Language Models are Biased." ICLR 2026 per a third-party summary [secondary only; verify]: visual encoders hold the information but the answer follows priors — [review](https://liner.com/review/vision-language-models-are-biased).
- Banerjee, Guntakala, Pike, Roongta, Yang, Li. Text Dominance in Paralinguistic Conflict. NeurIPS 2026 InterpScience Workshop accepted [PR, abstract]. Audio analogue: six audio LLMs, emotion among the cues; failure is not perceptual because speaker identity stays decodable at the language model's input — [arXiv](https://arxiv.org/abs/2610.25599).
- Mai, Nguyen, Takeda. arXiv Oct 2026 [PP, full text]: suggests label priors matter for VLM emotion on artworks, but this is untested, with no controlled separation — [arXiv](https://arxiv.org/html/2610.35244).
- Human precedent that makes this hard: Brooks and Freeman, Nature Human Behaviour 2018 [PR, abstract]. Using mouse-tracking and reverse correlation, conceptual similarity predicts perceived facial-emotion similarity, even controlling for physical similarity — [DOI](https://doi.org/10.1038/s41562-018-0376-6). A shared human-VLM confusion could be shared semantics, not shared perception.

**(d) Cue-level causal mediation inside a multimodal model: PARTLY. Method exists, facial-cue version not found.**
- Zhao, Wang, Liu, Tang. EMNLP 2026 accepted [PR, abstract]. Decomposes gender-cue effects into controlled indirect and direct effects by layer across six VLMs. Formally the design needed, on another domain — [arXiv](https://arxiv.org/abs/2609.16651).
- Deng et al. EMNLP 2026 accepted [PR, abstract]: steering across visual concept categories, entity localised vs abstract distributed — [arXiv](https://arxiv.org/abs/2605.05593). Song, Lepori, Pavlick [PP]: language recodes visual representations — [arXiv](https://arxiv.org/abs/2608.00035).
- No study located treats a facial cue or AU as the mediated variable.

**(e) Same information in VLMs and humans for ToM or RMET-like tasks: PARTLY. Same errors tested, same information not.**
- Strachan compares confusions and inversion, not which information is used. The gaze study (Zhang, Feng et al.) leaves its head-vs-eye hypothesis untested. GesBench is coarse and region-agnostic. Query x7 returned zero papers linking mechanistic VLM methods to human participants.

**(f) Inversion, composite, holistic in VLMs or ViTs: LARGELY OPEN, one partial scoop.**
- Inversion in a VLM: Strachan only (GPT-4o, RMET). No open-weight VLM located.
- Composite, part-whole, holistic or Thatcher in VLMs or ViTs: none located. CNN inversion is covered in Q2. MindSet includes Thatcherised stimuli and an updated evaluation set; I did not check whether ViTs were run on it, so the ViT cell may be less empty than it looks.

**Search log: queries with no relevant result (2026-11-04)**

arXiv API searches title, abstract, author and comment metadata only. Unless noted, each was ANDed with a VLM filter ("vision-language model(s)", "vision language model(s)", "multimodal LLM", "multimodal large language", "GPT-4V", "GPT-4o", "large vision language").

| Query (abbreviated) | Source | Hits | Relevant |
|---|---|---|---|
| VLM filter + "reverse correlation" | arXiv | 0 | 0 |
| VLM filter + "Bubbles" + psychophys* | arXiv | 0 | 0 |
| VLM filter + "classification image(s)" | arXiv | 13 | 0 (image-classification sense) |
| VLM filter + "diagnostic features/information" + face | arXiv | 0 | 0 |
| VLM filter + 3DMM/FLAME/blendshape/morphable + emotion | arXiv | 2 | 0 (generation tasks) |
| "reverse correlation" + DNN/transformer/LLM + face/expression, no VLM filter | arXiv | 0 | 0 |
| "reverse correlation" + LLM/VLM/GPT-4 | Europe PMC | 6 | 0 |
| "reverse correlation" + DNN + expression/emotion | Europe PMC | 11 | 1 (Daube, identity) |
| VLM filter + "face inversion"/"inverted faces"/"inversion effect" | arXiv | 2 | 1 (Strachan) |
| "face inversion" + ViT/LLM/GPT | Europe PMC | 3 | 0 direct |
| face inversion + transformer/ViT/CLIP/DNN, no VLM filter | arXiv | 19 | 0 |
| composite/holistic/Thatcher + transformer/ViT/DNN | arXiv | 11 | 0 for ViT |
| "composite face" + DNN/transformer | Europe PMC | 15 | 0 |
| Web: "CLIP upside-down faces inversion" | WebSearch | n/a | 0 |
| Web: "vision transformer holistic composite part-whole" | WebSearch | n/a | 0 |
| VLM filter + "error consistency"/"confusion matrices" + emotion + human | arXiv | 0 | 0 |
| FER + "error consistency"/"same mistakes", no VLM filter | arXiv | 1 | 0 |
| VLM filter + label semantics or language/knowledge prior + emotion | arXiv | 0 and 3 | 0 |
| semantic similarity + emotion + visual + confusion | arXiv | 0 | 0 |
| "counterfactual" + facial expression + VLM | arXiv | 0 | 0 |
| eye/mouth/upper/lower face region + VLM | arXiv | 1 | 0 |
| Emotion + image-free baseline + VLM | arXiv | 1 | 0 |
| mechanistic or steering terms + human participants/psychophysics + VLM | arXiv | 0 | 0 |
| Option order/position bias + emotion | arXiv | 15 | 0 |
| EU-Emotion + LLM | WebSearch | n/a | 1 (voice, gesture only) |
| RMET + Qwen/Gemma/open models | WebSearch | n/a | 0 |

Coverage limits:
- Google Scholar was not accessible.
- The Semantic Scholar API returned HTTP 429 and was not used.
- OpenReview and proceedings sites were reached only through web search.
- Full-text mentions inside papers are invisible to the arXiv API.
- A method may appear in a paper under different words, such as "perturbation-based saliency", which the logged queries could miss.

### Inferences
- A paper titled along the lines of "Mechanistic interpretability of emotion perception in VLMs" would be a late entry. Three papers since May 2026 cover it, and one is at ICML.
- All three share a limit: none tests against human perceptual data. The Zhao et al. caution about late-layer ESNs being answer selection, plus Gao et al., means even those results are open to the option-confound critique. Revisiting this is a contribution.
- Scoop speed is high. Many of the items above appeared in the last five months. A reverse-correlation study on a VLM is cheap to run, so I would treat ideas (a) and (f) as exposed for months, not years.

### Gaps
- Unpublished workshop posters at CCN, VSS and CogSci 2026 may contain Jack-style psychophysics on VLMs. These were not searched systematically.
- I could not assess the ICML 2026 paper's face content, because the fetched text does not state whether stimuli are faces.

## Q7. Novelty map

### Takeaway
Publishable novelty lies in Schyns's Degrees 2 and 3 for VLMs on faces. That means showing which facial information humans and models use under manipulated stimuli, then showing causally whether the model's internals carry the human-diagnostic features, separated from label semantics and answer machinery. Current Q1 and Q2 are necessary controls, not the contribution.

### Cited Findings

| Question | Status | Key papers | What specifically remains open |
|---|---|---|---|
| Can VLMs match humans on RMET accuracy? | answered | [Strachan](https://arxiv.org/abs/2410.22309); [Refoua 2026](https://www.nature.com/articles/s41598-026-39292-y); [Akben 2025](https://arxiv.org/abs/2508.08830) | Only whether scores survive option counterbalancing and contamination checks |
| Can VLMs recognise posed basic expressions? | answered | [Nelson 2025](https://www.nature.com/articles/s41746-025-01985-5); [FERBench](https://arxiv.org/abs/2511.00389); [EmoNet-Face](https://proceedings.nips.cc/paper_files/paper/2025/hash/8b2dd02001e72f65091115fb7f39d8c7-Abstract-Datasets_and_Benchmarks_Track.html) | Nothing at accuracy level |
| Do VLM emotion or social ratings share human structure? | answered at aggregate level; item level weak | [Asanuma 2025](https://arxiv.org/html/2505.12746); [Santavirta 2025](https://doi.org/10.1162/IMAG.a.134) | Whether structure comes from perception, semantics or shared statistics |
| Q1: Do models mix up the same mental-state pairs as humans? | partly | [Strachan](https://arxiv.org/abs/2410.22309); CNN analogue [Zhou 2022](https://doi.org/10.1126/sciadv.abj4383) | Open-weight VLMs; video; EU-Emotion; counterbalanced options; dissociating semantics |
| Q2: Is confusion pair-specific or generic difficulty? | partly | [Strachan](https://arxiv.org/abs/2410.22309) (information in errors); [Rajalingham](https://www.jneurosci.org/content/38/33/7255); [Geirhos 2020](https://arxiv.org/abs/2006.16736) | Trial-level error consistency on faces; needs options crossed with images, since fixed option sets partly determine confusion cells |
| Q3: Does the model causally use that structure? | partly (emotion identity), open (pair structure) | [Zhao 2026](https://arxiv.org/html/2608.17102); [Zhang ICML 2026](https://arxiv.org/html/2605.21980); [Xing 2026](https://arxiv.org/html/2609.34742) | Interventions that shift a specific human confusion pair and not others; none located |
| Do VLMs show a face inversion effect? | partly | [Strachan](https://arxiv.org/abs/2410.22309) | Open-weight models; whether inversion disrupts vision or the language interface; AU-level account of the qualitative error shift |
| Composite, part-whole, Thatcher in VLMs or ViTs? | open | CNN only: [Dobs 2023](https://doi.org/10.1073/pnas.2220642120); [Yovel 2023](https://doi.org/10.1098/rspb.2023.0093) | All of it; check MindSet ViT results first |
| Do VLMs use the same facial information as humans (reverse correlation, Bubbles, AU)? | open | Framework [Schyns 2022](https://eprints.gla.ac.uk/278852); identity CNN [Daube 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8515012); human [Jack 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3358835) | Any VLM classification image or AU-tuning curve; human-VLM comparison on the same stimuli |
| Same information for RMET-like mental states? | open | [Strachan](https://arxiv.org/abs/2410.22309); gaze [Zhang 2025](https://arxiv.org/html/2506.05412v1); [GesBench](https://arxiv.org/html/2610.15129) | Which cues carry mental-state labels in each system; head-vs-eye gaze test |
| Which internal components carry emotion in VLMs? | answered in part; crowded | [Zhang ICML 2026](https://arxiv.org/html/2605.21980); [Zhao 2026](https://arxiv.org/html/2608.17102); [Xing 2026](https://arxiv.org/html/2609.34742) | Whether these are perception, concept or answer machinery; complex mental states |
| Does internal emotion machinery line up with human cue use? | open | None located | The linking study: human-diagnostic cue as mediator |
| Perceptual evidence vs label semantics in emotion? | partly (general), open (emotion) | [Barua 2026](https://arxiv.org/html/2610.19407); [Liu ECCV 2026](https://arxiv.org/abs/2607.01503); human [Brooks 2018](https://doi.org/10.1038/s41562-018-0376-6) | Fitted evidence and bias for emotion; causal split at vision encoder vs decoder |
| Cue-level causal mediation in a VLM? | partly (method), open (facial cues) | [Zhao EMNLP 2026](https://arxiv.org/abs/2609.16651); [NOTICE](https://web3.arxiv.org/abs/2406.16320) | Direct and indirect effects of an AU or region on a mental-state judgment |
| Do MCQ formats contaminate activation-based emotion claims? | partly | [Gao 2026](https://arxiv.org/html/2608.22985); [Wiegreffe 2025](https://arxiv.org/abs/2407.15018); [Zheng 2024](https://arxiv.org/abs/2309.03882) | Cross-encoding audit of VLM emotion vectors; a free novel methodological contribution |
| Temporal: does video help mental-state reading? | open for faces | [AoT](https://arxiv.org/abs/2510.26241); [GesBench](https://arxiv.org/abs/2610.15129) | Static vs dynamic expression gain in VLMs against humans; no located study |

### Inferences
Ranked by my judgment of novelty and defensibility:
1. **Same-stimulus human and VLM psychophysics, followed by causal mediation.** Use parametric or AU-controlled faces. Estimate per-cue tuning in humans and models, then test by intervention whether the internal pathway carries each human-diagnostic cue. This answers the supervisor's "HOW" at Degrees 2 and 3 and fills the empty cell in the map.
2. **Decomposing a shared confusion into three sources:** perceptual cue overlap, label semantics, and answer machinery, as a factorial design on image, option set and readout. This turns Q1 to Q3 into controls, and it gives the study a claim that prior work cannot make.
3. **An option-confound audit of VLM emotion interpretability.** Cheap, publishable as a methods result, and it protects the main paper. The downside is that it is a critique rather than a discovery.
4. **Inversion as a causal probe.** Testing whether Strachan's qualitative error shift arises in the vision encoder or in the decoder would extend the one existing scoop. Note the caution that Strachan's result is from a closed model with fixed options, and may not replicate.

Skeptical points against this map:
- Null evidence here is metadata-level, from one day.
- Parametric faces lack ecological validity, and EU-Emotion and RMET lack parametric control. The study may need both.
- Open VLMs may be too weak on subtle mental states to yield interpretable tuning; pilot first. Claude 3 Opus was near chance on RMET in Refoua et al., showing model capability varies widely.
- Reverse correlation needs many trials per model. That is cheap locally, but it requires human comparison data that I have not confirmed is available.

### Gaps
- Whether existing human reverse-correlation data for complex mental states exist at all. The located work covers basic emotions, and new human data may be required.
- Whether any 2026 VSS or CCN abstract already pursues rank 1. This needs direct checking before committing.
- The ICML 2026 (Zhang) and CMES final sections were not fully read, and the last ~28K and ~49K characters respectively went unread.
