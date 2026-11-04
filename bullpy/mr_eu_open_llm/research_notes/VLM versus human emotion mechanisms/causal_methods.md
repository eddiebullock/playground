# Causal methods for testing which facial information a VLM uses for mental-state recognition

Scope note. Date of search: 2026-11-04. Licence and repository facts were read from the GitHub API, raw LICENSE files, Hugging Face configs or transformers source on that date. Model-architecture statements marked "config" or "code" come from those primary files, read as excerpts, and were not run. Numbers under Inferences that are marked "computed" were calculated for these notes (scipy, exact binomial and Clopper-Pearson). Project facts come from local files in this repo and are marked "project file". Anything recalled but not located is in Gaps, not in Cited Findings.

Corrections to the brief, found during location:
- Palit et al. studied BLIP, not LLaVA. LLaVA tracing is in Neo et al., NOTICE, FCCT and "Where To Look?".
- FaReT extends MakeHuman, not FaceGen.
- "Causal Abstractions of Neural Networks" is Geiger, Lu, Icard, Potts. Wu is on the later IIT paper.
- No work matching "Hewitt's alignment" or a "Weiskopf" interpretability method was found in three searches. Do not cite these until someone supplies a title.
- Gemma 4 E4B does accept video, per the transformers docs, despite third-party tables saying otherwise.

## 1. Stimulus intervention tools: what can edit a cue on real video, and how to stop the generator being the confound

### Takeaway
No located tool gives verified, AU-level, identity-preserving editing of real video. The best free option is a two-arm design: a parametric arm where AU values are known by construction (FACSHuman), and a photoreal arm (LivePortrait, MagicFace per frame) checked by independent AU detectors. Every edited clip must be compared with a null-edited clip passed through the same generator, never with the original.

### Cited Findings

Licences and status (GitHub API and LICENSE files, read 2026-11-04):

| Tool | Control | Licence | Video? | Status and evidence |
|---|---|---|---|---|
| LivePortrait | Implicit keypoints, stitching, 2 retargeting modules | Code MIT, but [InsightFace models non-commercial research only](https://github.com/KlingAIResearch/LivePortrait/blob/main/LICENSE); research use OK | Yes, [v2v editing added July 2024](https://huggingface.co/KwaiVGI/LivePortrait/raw/main/README.md) | [12.8 ms per frame on RTX 4090](https://arxiv.org/abs/2407.03168). Repo redirected to KlingAIResearch, pushed 2026-06 |
| MagicFace | [AU values directly, plus ID encoder and attribute controller](https://arxiv.org/pdf/2501.02260) | [Apache-2.0](https://github.com/weimengting/MagicFace) | Image-only evaluation | Closest to true AU control. Pushed 2025-11 |
| FACSHuman | [Almost all FACS AUs, MakeHuman plugin](https://link.springer.com/10.3758/s13428-021-01559-9) | [LGPL-3.0](https://github.com/montybot/FACSHuman) | Animated synthetic | [Four validation studies reported](https://link.springer.com/10.3758/s13428-021-01559-9); results not read. Repo labelled v0.0.1 alpha |
| FaReT | [MakeHuman plugins; 27 identities, 6 expression poses](https://link.springer.com/article/10.3758/s13428-020-01421-4) | [GPL-3.0](https://github.com/fsotoc/FaReT) | Dynamic rendering | [Small human validation](https://link.springer.com/article/10.3758/s13428-020-01421-4) |
| Generative Face Grammar | [42 core AUs randomly combined on photorealistic identities](https://pmc.ncbi.nlm.nih.gov/articles/PMC4927208) | No public release located | Yes (animation) | Strongest human reverse-correlation record. See Gaps |
| EMOCA / DECA | FLAME 3DMM parameters | [MPI non-commercial scientific](https://github.com/radekd91/emoca/blob/master/LICENSE) ([DECA same](https://github.com/yfeng95/DECA/blob/master/LICENSE)) | Frame-wise | Research OK |
| SMIRK | FLAME via neural renderer | [MIT repo](https://github.com/georgeretsi/smirk); FLAME terms not read | Frame-wise | [1.28 mm mean vertex error on MultiFace; user study favoured it](https://arxiv.org/pdf/2404.04104). No AU evaluation found |
| HunyuanPortrait | Driven animation | [Academic, research, education only](https://github.com/Tencent-Hunyuan/HunyuanPortrait/blob/main/LICENSE) | Yes | Pushed 2025-12 |
| Arc2Face | Identity-conditioned generation | [MIT](https://github.com/foivospar/Arc2Face) | Not confirmed | Expression adapter not inspected |
| DiffusionRig | 3DMM-guided | [Adobe research, non-commercial](https://github.com/adobe-research/diffusion-rig/blob/main/LICENSE.md) | Not confirmed | Pushed 2023 |
| GANimation | AU vector | [GPL-3.0](https://github.com/albertpumarola/GANimation) | Image | Last push 2019 |
| ICFace | Not inspected | [No standard licence detected; no LICENSE at root](https://github.com/Blade6570/icface) | Not confirmed | Last push 2020 |
| Pix2Pix-zero | Text-direction edit | [MIT](https://github.com/pix2pixzero/pix2pix-zero) | Image | No AU control |
| face-vid2vid | Keypoints | [NVlabs repo: no licence detected](https://github.com/NVlabs/face-vid2vid); [unofficial reimplementation: none](https://github.com/zhengkw18/face-vid2vid) | Yes | Avoid; LivePortrait covers the same niche |
| Wan 2.2 | Generative video | [Apache-2.0](https://github.com/Wan-Video/Wan2.2) | Yes | Regenerates pixels; AU fidelity not found |
| Stable Video | Generative video | [Code MIT](https://github.com/Stability-AI/generative-models); weight terms not read | Yes | AU fidelity not found |
| VASA-1 | Audio-driven | No public code at the expected repo path ([404](https://github.com/microsoft/VASA-1)); [project page](https://www.microsoft.com/en-us/research/project/vasa-1/) as fetched says nothing about release | n/a | Treat as closed; see Gaps |

Gaze and pose:
- [GazeGaussian (ICCV 2025)](https://arxiv.org/abs/2411.12981v2) models face and eyes in two streams with an eye rotation field and handles head pose. The [repo is public, pushed 2026-10, no licence detected](https://github.com/ucwxb/GazeGaussian). It needs per-subject fitting, I expect, which I did not confirm.
- [ETH-XGaze code is CC BY-NC-SA 4.0](https://github.com/xucong-zhang/ETH-XGaze). [Gaze360 is a research-only licence, repo archived, sharing limited to colleagues at the same institution](https://github.com/erkil1452/gaze360/blob/master/LICENSE.md). Both are useful for verifying gaze, not for editing it.

Verification and perceptual validity:
- The search found no paper evaluating LivePortrait at the AU level. Recent papers benchmark it with [Average Expression Distance](https://arxiv.org/pdf/2507.12956) or [CSIM and APD](https://arxiv.org/pdf/2512.15524), which are proxies. A [10-person user study of expression and identity consistency](https://arxiv.org/pdf/2503.19906) included it, with mixed results.
- AU consistency has precedent as a reenactment metric: [One-shot Face Reenactment](https://arxiv.org/pdf/1908.03251) reports it with pose consistency and identity, and [ReenactGAN](https://arxiv.org/pdf/1807.11079) correlated detector responses between source and output.
- Pixel metrics are a poor fit for reenactment, per [MagicPortrait](https://arxiv.org/pdf/2504.21497).
- Human validity: [Becker et al. 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11362322/) found videos and deepfakes rated similarly in intensity, while dynamic morphs were rated less intense than videos (all emotions) and deepfakes (fearful, happy, sad). Their deepfakes were DeepFaceLab puppetry.
- Belief alone matters: [faces presumed to be deepfakes gave dampened responses for smiles but not angry expressions](https://www.nature.com/articles/s41598-023-42802-x.pdf).
- [Morphs were judged less genuine for happiness and sadness, not anger or fear](https://link.springer.com/10.3758/s13428-024-02443-y).
- Reenactment [changed conveyed emotion without much changing personality judgments](https://graphics.tu-bs.de/publications/groth2020automatic) (thesis).
- [Humans detect deepfakes holistically, and blending, texture and movement discontinuities are known cues](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11362322/). A ViT may pick up the same cues.
- [FaceGen-style images give control but do not always capture natural appearance](https://www.arxiv.org/abs/1901.09447) (review; attribution of URL to that passage from search snippet, check).

Confound precedent:
- [CaCE](https://arxiv.org/pdf/1907.07165) uses generative counterfactuals precisely because correlated concepts mislead. Ground-truth CaCE needs access to the generation process, which is what a parametric arm provides.

### Inferences
- Primary contrast: `y(edit_k(x)) - y(edit_0(x))`, where `edit_0` is a zero-delta pass through the same generator. This cancels generator artifacts, re-encoding, and face-crop effects at first order. Comparing with the original measures the generator as much as the cue.
- Run a generator-detection control. Train a probe on image tokens to classify `edit_0` versus original. If it succeeds easily, the generator is visible to the model. That does not void the contrast above, but it warns of an interaction.
- Require a bidirectional edit where possible (add AU12 to a neutral clip, remove it from a smile clip). Symmetric effects are harder to explain by artifact.
- Use dose levels (for example 0, 25, 50, 75, 100 percent of AU delta). A generator artifact tends to be step-like at any non-zero edit; a cue effect should be graded. This is a heuristic, not a guarantee.
- Identity: for each edit, record ArcFace cosine with `edit_0`, and include it as a covariate. Note that a large expression change lowers identity similarity in many embedders, so a fixed threshold can exclude the strongest edits.
- Cross-generator replication is the strongest guard. An effect that appears in FACSHuman (known AU, no real texture) and in LivePortrait (real texture, inferred AU) cannot be blamed on one generator's artifact.
- LivePortrait does not expose AUs. Its control is implicit keypoints plus eye and lip retargeting. An AU-to-keypoint mapping would have to be learned, for instance by regressing keypoint deltas on AU changes from driving clips, then verified. Budget a week for this.
- MagicFace per frame will probably flicker because it was evaluated on images only. Optical-flow smoothing adds another artifact, so a per-frame arm is best suited to single frames or composite grids.
- Project stimuli are composite grids of 4 to 16 frames (project file `research/COMPUTE.md`). This makes single-frame editors more usable, but see section 7 on token resolution.
- Do not regenerate pixels with Wan or Stable Video for causal work. They resynthesise the whole frame, so background, skin texture and identity change with the edit.

### Gaps
- No located public release of the Generative Face Grammar. Only the [2012 paper](https://researchportal.port.ac.uk/en/publications/perception-driven-facial-expression-synthesis/) and later use were found. Assume it needs a collaboration with Glasgow.
- [PMC3358835](https://pmc.ncbi.nlm.nih.gov/articles/PMC3358835) could not be fetched, so trial counts per observer for Glasgow work are unverified. I recall thousands per observer; not confirmed.
- Whether VASA-1 was ever released: the fetched page is silent. My recollection that Microsoft declined to release it is not verified.
- FACSHuman validation effect sizes were not read.
- No study located on whether VLM vision encoders detect LivePortrait or MagicFace artifacts.
- FLAME model licence not read. Blender-based AU rigs not researched.
- Arc2Face expression adapter, DiffusionRig, ICFace and Stable Video weights were checked for licence only.

## 2. AU detection for verifying edits

### Takeaway
Use LibreFace (best documented intensity, research licence) plus OpenFace 3.0 or ME-GraphAU as a second opinion, and treat a manipulation check as a within-clip change, not an absolute score. Published F1 of 50 to 66 is too low for trusting one detector on one clip, and accuracy on generated faces is unmeasured.

### Cited Findings

Same-paper comparison, [OpenFace 3.0 Tables III and IV](https://arxiv.org/html/2506.02891):

| Tool | DISFA F1 | BP4D F1 | CPU ms/frame | Licence |
|---|---|---|---|---|
| ME-GraphAU (labelled SOTA) | 66 | 66 | n/a | [MIT](https://github.com/CVI-SZU/ME-GraphAU) |
| LibreFace | 61 | 62 | 33 | [USC: educational, research, non-profit](https://github.com/ihp-lab/LibreFace/blob/main/LICENSE.rst) |
| OpenFace 3.0 (multitask) | 60 | 62 | 38 | [CMU academic non-commercial](https://github.com/CMU-MultiComp-Lab/OpenFace-3.0/blob/main/LICENSE) |
| Py-Feat | 54 | 55 | 82 | [Code MIT; weights by family](https://github.com/cosanlab/py-feat/blob/main/LICENSE) |
| OpenFace 2.0 | 50 | 53 | 75 | [Academic non-commercial research only](https://github.com/TadasBaltrusaitis/OpenFace/blob/master/OpenFace-license.txt) |

- These are numbers reported by the OpenFace 3.0 authors. [LibreFace reports BP4D 62.0](https://arxiv.org/html/2308.10713v2), which matches.
- Intensity: [LibreFace DISFA PCC 0.63 vs OpenFace 2.0 0.59](https://arxiv.org/html/2308.10713v2). Per-AU PCC: AU1 .63, AU2 .72, AU4 .78, AU5 .59, AU6 .59, AU9 .62, AU12 .85, AU15 .36, AU17 .50, AU20 .34, AU25 .94, AU26 .68. ICC not reported.
- LibreFace covers 17 AUs: [12 by intensity, 5 by occurrence (AU7, 10, 14, 23, 24)](https://arxiv.org/html/2308.10713v2).
- OpenFace 3.0 reports [no intensity accuracy, and no quantitative video-level or occlusion claims for AUs](https://arxiv.org/html/2506.02891).
- JAA-Net: [secondary summary gives BP4D 60.0, DISFA 56.0](https://www.emergentmind.com/papers/1803.05588), unverified against tables. On DISFA it uses [8 AUs](https://arxiv.org/pdf/1803.05588). [No licence detected](https://github.com/ZhiwenShao/PyTorch-JAANet), so it is best avoided.
- Py-Feat: [AU and emotion detectors are affected by occlusion of specific structures; the XGB AU model is fairly robust to rotation except at larger angles](https://arxiv.org/abs/2104.03509v4). The [ArcFace identity weights inherit InsightFace non-commercial-research terms](https://github.com/cosanlab/py-feat/blob/main/LICENSE). The licence file says an MIT implementation licence does not settle per-checkpoint permissions.
- MediaPipe: [Apache-2.0](https://github.com/google-ai-edge/mediapipe), [52 ARKit-style blendshapes](https://arxiv.org/html/2501.13432v2). A [2026 paper is described as one of the first documented mappings to AUs](https://iris.cnr.it/handle/20.500.14243/583461). No validation against FACS coding was found.
- [Accuracy on one stimulus type does not guarantee accuracy on another, and is higher for posed than spontaneous expressions](https://pmc.ncbi.nlm.nih.gov/articles/PMC8235167).
- Cross-dataset generalisation is [among the hardest problems in the area](https://arxiv.org/pdf/2103.02484). A [2026 preprint proposes leave-one-dataset-out, exposing instability hidden by single-dataset cross-validation](https://arxiv.org/html/2604.02162v1).

### Inferences
- OpenFace 2 and 3 and LibreFace licences all allow academic noncommercial research, which fits this study. Read their publication and citation terms. OpenFace 2 [requires citation](https://github.com/TadasBaltrusaitis/OpenFace/blob/master/Copyright.txt).
- The target AUs differ in measurability. AU12 (.85) and AU4 (.78) are well-covered in LibreFace. AU15 (.36) and AU20 (.34) are not, so claims for them need FACSHuman ground truth.
- Manipulation-check rule: an edit passes if at least two detectors from different lineages show the intended AU delta on target frames, off-target AUs stay within a pre-set band relative to `edit_0`, and identity and pose stay in band. Calibrate bands on `edit_0` test-retest.
- Calibrate detectors on the parametric arm: render a known AU ramp in FACSHuman and fit detector output against ground truth, which gives a dose-response per detector and AU. Synthetic faces are out of training distribution, so this bounds the detector rather than validating it.
- Gaze and head yaw are measured by OpenFace-family tools; I did not extract accuracy.

### Gaps
- No published accuracy of any AU detector on LivePortrait or diffusion-edited faces.
- ME-GraphAU intensity capability not confirmed. The OpenFace 3.0 table labels it SOTA; the mapping of "Luo et al." in LibreFace to ME-GraphAU is my reading, not stated.
- No DISFA or BP4D result for MediaPipe located.
- OpenFace 3.0 AU list not stated in the paper.

## 3. Occlusion and information-based methods

### Takeaway
Pixel-space Bubbles and region-Shapley are feasible on a VLM because the model is deterministic and one forward pass costs seconds. A coalition design over 8 to 12 FACS regions is cheaper and easier to interpret than free-form masks. Token-space masking is far cheaper but answers a different question and has spatial leakage problems.

### Cited Findings
- [Bubbles (Gosselin and Schyns 2001)](https://data.zhenzhi.com/details/1/887145128) attributes performance to sampled information across gender, expressiveness and identity tasks.
- [Smith et al. 2005](https://typeset.io/pdf/transmitting-and-decoding-facial-expressions-1f2ke2ckt7.pdf) applied Bubbles across 5 one-octave spatial-frequency bands to 70 CAFE images of 6 basic expressions plus neutral. A later paper notes [different parts of the face transmit different expressions](https://pmc.ncbi.nlm.nih.gov/articles/PMC2680487).
- A [Politecnico di Torino thesis](https://webthesis.biblio.polito.it/34885) compared deep models with human eye tracking on expressions: Bubbles gave the closest match to canonical human saliency, but models and humans generally did not share viewing patterns (thesis only).
- [Xu et al. 2018](https://mlanthology.org/cvprw/2018/xu2018cvprw-using) used Bubbles and RSA on a face-identity ResNet.
- Reverse-correlation trial counts: [about 5,000 to 10,000 in typical psychophysics](https://visionsciences.org/presentation/?id=1990); [under 1,000 with noise tailored to the base image's spectrum](https://www.visionsciences.org/presentation/?id=3401) (posters); a [DCNN face-gender study used 20,000 noise trials](https://pmc.ncbi.nlm.nih.gov/articles/PMC7870475); [Brief-RC reduces trials](https://perso.uclouvain.be/vincent.yzerbyt/Schmitz%20et%20al.%20EJSP%202024.pdf).
- [RISE](https://arxiv.org/pdf/1806.07421) weights random masks by output probability and is black box. [Deletion and insertion metrics](https://arxiv.org/pdf/1806.07421) are its evaluation.
- [PixelSHAP](https://arxiv.org/html/2503.06670v1) gives Shapley values on segmented objects, tested on GPT-4o, Gemini-2.0-flash, LLaVA-v1.5-7B, LLaMA-3.2-11B-Vision. It needs only marginally more queries than objects with first-order masking and takes about 15 to 74 s per image. Limits: segmentation dependence, silhouettes left by exact masks, neighbours hidden by box masks, and it scores by text-embedding similarity.
- [FIxLIP](https://proceedings.neurips.cc/paper_files/paper/2025/file/4e7686b7b147fac3352b24ab86ff78fb-Paper-Conference.pdf) improves Shapley-interaction efficiency 5 to 20 times, for encoders such as CLIP and SigLIP-2, not generative VLMs.
- A [EACL 2026 paper](https://preview.aclanthology.org/ingest-eacl/2026.eacl-long.210.pdf) uses KernelSHAP-style patches inside boxes for VLM hallucination and bias.
- Token-space ablation precedent: [Neo et al.](https://arxiv.org/html/2410.07149v2) on LLaVA 1.5: object tokens only (12.6) gave a 33 percent drop in generative identification, object plus 1 buffer (33.4) gave 72 percent, random 40 tokens 2.4 percent. So information spreads beyond the object footprint.
- [Where To Look?](https://arxiv.org/html/2608.10758v1) reports near-zero correlation between causal contribution and spatial overlap in the vision encoder (Qwen-2.5 0.09 DDPM, 0.03 blur; LLaVA-NeXT sign flips between corruptions). It is a preprint with undefined metrics and no code, and its score for generative VLMs is unclear.

### Inferences
- Query budget, assuming 1 to 3 s per forward pass (assumption, time one smoke job). A 12 h job gives about 14,000 to 43,000 forwards per GPU.
- Exact Shapley over 8 regions (brow, eyes, nose, cheeks, upper lip, mouth corners, chin, forehead or gaze) needs 256 coalitions. For 40 items and 3 models this is 30,720 forwards, roughly 9 to 26 GPU-hours (computed). This also yields all pairwise interactions, which test redundancy at the stimulus level.
- Bubbles: 2,000 masks per item across 30 items and one model is 60,000 forwards, 17 to 50 GPU-hours (computed). Run one model first and stop when split-half reliability of the map plateaus.
- The model's response is deterministic, so all sampling noise comes from masks. Fewer trials than human studies may suffice, but this should be shown by the convergence curve, not assumed. No VLM-specific trial study was found.
- Occluder choice is itself an intervention. A grey disc adds an edge and a "something is covering the face" percept, and a VLM may report that. Prefer replacing the region with the same region from the actor's neutral frame, then fall back to blur. Compare at least two fills.
- Time dimension: mask frames as well as regions (early, peak, late) for dynamics. In composite grids, swap individual tiles.
- Use logit difference or logprobs over the label set rather than text-embedding similarity. Embedding similarity would be dominated by wording.
- Feature visualisation and MEIs through a VLM would be expensive and hard to validate for facial cues. Treat them as hypothesis generators only. I recommend skipping them for this timeline.

### Gaps
- RISE mask counts could not be extracted. I recall thousands per image; unverified.
- No application of Bubbles to a generative VLM was found, and no video-Shapley for faces. No dynamic Bubbles study located.
- MEI literature was not located in this pass (recalled: Walker et al. inception loops). Nor were ROAR or Covert's removal framework.
- No published forward-pass cost for these three models. Project estimates are planning figures only (project file).

## 4. Internal causal inference: what the literature supports, and the failures

### Takeaway
Causal tracing through image tokens to answers exists for LLaVA, BLIP, Qwen-VL and DeepSeek-VL, mostly on object tasks, and one paper touches facial expression. Nothing located does a cue-specific mediation test on faces. Null ablation has several known explanations (backup paths, self-repair, overdetermination, interactions, wrong site), and two of the three study models have architectural bypasses that a standard image-token patch would miss.

### Cited Findings

Architecture facts that change what a patch does:
- **Qwen3-VL-8B** ([config](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/raw/main/config.json), [code](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)): 27 ViT layers, 36 LM layers, patch 16, spatial merge 2, temporal patch 2, `deepstack_visual_indexes` [8, 16, 24]. `_deepstack_process` adds merged features from those ViT layers to hidden states at image positions, in LM layers indexed 0, 1, 2. The [technical report](https://arxiv.org/pdf/2511.21631v1) describes injection into corresponding LLM layers.
- **Gemma 4 E4B** ([config](https://huggingface.co/google/gemma-4-E4B-it/raw/main/config.json), [docs](https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/model_doc/gemma4.md), [code](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma4/modeling_gemma4.py)): 16 ViT layers, 42 LM layers, patch 16 with pooling 3, default 280 soft tokens. `hidden_size_per_layer_input` is 256, so Per-Layer Embeddings are active. A gated signal derived from `inputs_embeds` is added at the end of every decoder layer. Licence [Apache-2.0](https://huggingface.co/google/gemma-4-E4B).
- **Molmo2-O-7B** ([config](https://huggingface.co/allenai/Molmo2-O-7B/raw/main/config.json), [code](https://huggingface.co/allenai/Molmo2-O-7B/raw/main/modeling_molmo2.py)): 27 ViT layers, 32 LM layers, patch 14. The adapter concatenates ViT layers -3 and -9, then applies attention pooling and a projector. A mask function gives bidirectional attention within image tokens. Custom remote code. Licence Apache-2.0 on the Hugging Face tag. Video pooling 3 by 3 appears only in an [unofficial summary](https://books-1k9.pages.dev/en/molmo2/overview).

VLM causal tracing:
- [Palit et al. 2023](https://arxiv.org/abs/2308.14179): adapted causal tracing to BLIP by perturbing image embeddings; later layers causally relevant. [Code, MIT](https://github.com/vedantpalit/Towards-Vision-Language-Mechanistic-Interpretability).
- [NOTICE](https://arxiv.org/html/2406.16320v3): Semantic Image Pairs replace Gaussian noise, with clean and corrupt images differing in one semantic property. Includes FER-2013 (48x48 grayscale; BLIP 66.1 percent, LLaVA-1.5 79.31 percent; two-option prompts with position balanced). Pairs drew on opposing emotions. For LLaVA, the "Assistant:" token was patched because patching the answer token had tiny effects (max 0.016). Head L3.H0 linked the emotion token to the mouth region. FER-specific head statistics are not separate. [Repo](https://github.com/wrudman/NOTICE) has no licence detected.
- [Neo et al. ICLR 2025](https://arxiv.org/html/2410.07149v2): mean ablation with the mean visual token over 50,000 ImageNet images; attention knockout by mask to negative infinity over layer windows. Blocking object tokens to the last token in mid-late layers gives 0.88 relative probability; blocking all non-object visual tokens gives 0.67. Blocking visual tokens to the last visual row had no effect.
- [FCCT](https://arxiv.org/html/2511.05923): LLaVA-1.5 and Qwen-VL-Chat, 500 COCO images, yes-or-no prompts, Gaussian noise on the whole image, restores one component at one layer for seven token categories, Recovery Rate as metric. Last-token MHSA in intermediate layers has strong causal effect. No code located. Corruption covers the whole image, so it does not isolate visual tokens.
- [Where To Look?](https://arxiv.org/html/2608.10758v1): patching inside vision encoders, finds causal tokens outside target regions (details in section 3).
- [ICLR 2025 blog](https://d2jud02ci9yv69.cloudfront.net/2025-04-28-vlm-understanding-29/blog/vlm-understanding/) secondary reports: Basu et al. LLaVA retrieves visual information mainly in layers 1 to 4; Palit image effects at BLIP layer 11 of encoder and 9 to 11 of decoder; neuron deactivation in VLMs often had minimal effect.
- [Viewpoint rotation study](https://arxiv.org/html/2604.15294) applied path patching, head knockout and selective fine-tuning to Qwen2.5-VL-7B, but on text-only input, so it is no help for vision paths.

Methods:
- Path patching: [Goldowsky-Dill et al. 2023](https://arxiv.org/abs/2304.05969) tests whether a behaviour is localised to a set of paths; open-source framework released.
- Causal scrubbing: [Redwood post](https://lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing) resamples activations that a hypothesis says should not matter, then checks behaviour is preserved. Tested on small transformers.
- Causal abstraction: [Geiger et al. 2021](https://papers.nips.cc/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html) interchange interventions; [JMLR theory paper](http://jmlr.org/papers/volume26/23-0058/23-0058.pdf) unifies patching and mediation; [IIT](https://mlanthology.org/icml/2022/geiger2022icml-inducing) trains models and so changes them.
- [DAS](https://arxiv.org/pdf/2303.02536v4.pdf) finds alignments by gradient descent; [pyvene, Apache-2.0](https://github.com/stanfordnlp/pyvene) runs interventions on PyTorch modules. A [rebuttal to the illusions critique exists](https://www.arxiv.org/pdf/2401.12631). [CausalGym found DAS beat linear probing on causal tasks](https://hyper.ai/ar/papers/acl__2024__2024.acl-long.785) (summary).
- LEACE: [closed-form, stops all linear classifiers detecting the concept, smallest such edit in the least-squares sense](https://arxiv.org/pdf/2306.03819). Concept scrubbing applies it at every layer. [Code MIT](https://github.com/EleutherAI/concept-erasure). Linear only. [LEOPARD adds density-matching nonlinear erasure](https://arxiv.org/pdf/2507.12341). A report that [LEACE returns a one-dimensional subspace holding about half the concept information](https://arxiv.org/pdf/2307.15054), with mean erasure ratio 0.64, appears to come from "A Geometric Notion of Causal Probing"; two search summaries disagreed on the origin.
- TCAV: [CaCE shows correlated concepts mislead TCAV](https://arxiv.org/pdf/1907.07165); [Bahadori and Heckerman](https://arxiv.org/pdf/2007.11500v3) debias with a causal prior.
- SAE: [SAE-V on CLIP and DINOv2](https://arxiv.org/html/2502.06755) validates causal features in vision models. [Exploratory LLaVA SAE work exists](https://arxiv.org/html/2508.11277). [VISTA](https://arxiv.org/html/2605.24946v1) maps image tokens into a text SAE but retrains the projector at 200 to 290 A100 GPU-hours on 595k plus 665k examples, and was tested on Gemma 2 and LLaMA 3.1, not Qwen.
- Causality framing: a [Feb 2026 paper](https://arxiv.org/abs/2602.16698?context=cs) argues Pearl's hierarchy limits what interpretability studies can justify. [Bucketing the Good Apples](https://www.emergentmind.com/papers/2605.02234) finds input subspaces where an interpretation is faithful.
- [Mediator survey](https://arxiv.org/html/2408.01416v3): verifying a hypothesis prioritises sparsity, generality, selectivity and interchange accuracy; use layers to narrow search first; probes were often not causally verified.

Known failures:
- Probes are not causal use: [high accuracy shows correlation, and results depend on task design](https://d2jud02ci9yv69.cloudfront.net/2025-04-28-vlm-understanding-29/blog/vlm-understanding/). This matches the project audit.
- Backups: [IOI found eight backup name-mover heads that activate only when primaries are ablated](https://arxiv.org/pdf/2211.00593.pdf).
- Self-repair: [Rushing and Nanda](https://arxiv.org/abs/2402.15390) found repair across model families, with the final LayerNorm repairing up to 30 percent of direct effect. Avoid zero ablation.
- Overdetermination: [Mueller 2024](https://arxiv.org/abs/2407.04690): when several components each suffice, ablating one misses all of them. Effects are also non-transitive. Remedies: diverse data, positive and negative counterfactuals.
- Hydra effect: [the survey notes downstream components may matter only after upstream ones are removed](https://arxiv.org/html/2408.01416v3).
- Interactions: [Vaidyanathan et al. 2026](https://arxiv.org/html/2606.27510): patching estimates NIE = PIE + INT. On IOI with pABC corruption, summed PIEs predicted +5.78, the circuit gave 1.52, and summed cross-interaction was -4.05. Rank correlation of NIE and PIE dropped to 0.51 under pABC but 0.92 under symmetric swap. INT is a diagnostic; no corrective estimator exists. Code link stated in the paper [returned 404 on the API](https://github.com/sankaranv/mech-interp-mediation-analysis).
- Divergence: [Grant et al. 2025](https://arxiv.org/html/2511.04638v4) separate harmless from pernicious divergence (hidden pathways, dormant changes). A null patch can still have a causal role in other contexts. Report divergence and test context sensitivity. Metrics: EMD, local PCA distance, KDE density.
- Illusions: [Makelov et al.](https://chatpaper.com/paper/5930) (seen via search summary only) show a direction can look causal via a dormant pathway; removing the null-space component is a control.
- Corruption sensitivity: [Zhang and Nanda](https://arxiv.org/html/2309.16042) prefer symmetric, in-distribution corruption, logit difference over probability, single-layer before sliding window, and checking across settings. The choice of corrupted token can change findings.
- Steering: [Tan et al.](https://arxiv.org/html/2407.12404v8) report strong input variation, spurious biases and brittleness to prompt changes. A [critique](https://www.lesswrong.com/posts/QQP4nq7TXg89CJGBh/a-sober-look-at-steering-vectors-for-llms) says steering often degrades capability.
- SAE: [DeepMind deprioritised SAE research after they lost to linear probes on a downstream task](https://www.longtermwiki.com/wiki/E480/data) (secondary). [Sanity Checks 2026](https://arxiv.org/html/2602.14111) argues SAEs do not reliably decompose internal mechanisms.

### Inferences

Why the 5/180 null is uninformative about vision: it intervened on a last-token axis, a site the audit says carries option-list text. Literature on nulls adds that it may also be under-dosed, redundant or repaired. It says nothing about the vision path.

Patch cut-points per model. These follow from the code above and need a unit test:

| Claim tested | Qwen3-VL-8B | Gemma 4 E4B | Molmo2-O-7B |
|---|---|---|---|
| Vision tower | ViT block outputs; 8, 16, 24 also feed DeepStack | ViT block outputs | ViT blocks; -3 and -9 are the read points |
| Image tokens entering LM | `inputs_embeds` at image positions AND all 3 `deepstack_visual_embeds`. Patching only `inputs_embeds` leaves the injection | `inputs_embeds` at image positions, which should also change the per-layer projection. Confirm by hook | Post-projector tokens |
| Complete vision cut in LM | Image positions after LM layer index 2 | No single-site cut if per-layer input stays live; must also patch `per_layer_inputs` at image positions | Image positions at any layer |
| Read path to answer | Attention mask text queries to image keys, per layer window | Same; check attention type per layer | Same |

> **Measured correction, 2026-11-04 (vision gates, cv2 4.11; `research/DECISIONS.md`).** The table above is the pre-run reading of the code. Measured: Gemma needs both routes cut at L0, and from L1 a mean of 42 percent of the image effect remains. Gemma layers 24 to 41 reuse keys and values from layers 22 and 23, so image-position patches at L24 or later are guaranteed false nulls. Gemma used 253 to 256 soft tokens here, not 280. Molmo "image positions at any layer" is wrong: a cut from L1 leaves a mean of 11.7 percent. Molmo vision features are added to the patch embedding, and its bidirectional image attention does not apply because `token_type_ids` is removed. Qwen "after LM layer index 2" is wrong on a provisional 2-clip smoke: 31 percent remains at bfloat16 and 46 percent at float32. The Qwen full gate is pending.

- The DeepStack is a direct redundancy threat: if only post-merger tokens are patched, three ViT depths still reach the LM. This is a candidate explanation for any future Qwen null.
- Gemma's per-layer path means a residual patch at layer k may be partly re-injected downstream. I read this from excerpts, and how image positions behave depends on lines I did not fully trace.
- Estimand: define with counterfactual pair `(x_on, x_off)`. Total effect is `y(x_on) - y(x_off)`. Necessity at site S: run `x_on` with S taken from `x_off`. Sufficiency at S: run `x_off` with S taken from `x_on`. Report both, because the difference between them estimates the interaction term that a one-way patch hides. Use Recovery Rate, with the denominator guarded when total effect is small.
- Mediation claim for cue C through representation R requires all of: (1) C changes the answer at stimulus level; (2) C changes R, as a within-pair difference; (3) patching R from `x_off` into `x_on` reduces the effect; (4) patching R from `x_on` into `x_off` restores it; (5) the patch stays near-manifold; (6) patching a control representation of similar norm does nothing; (7) the result replicates in a second generator. No located paper does all seven in a VLM.
- Layered search, per the survey's advice. Start with coarse complete cuts: all image tokens, all text-to-image attention. Bisect by layer window. Then go to region tokens, then subspaces. A bisection costs about log2(36) = 5 to 6 steps per model rather than a full heat map.
- Redundancy test: for sites A and B, interaction contrast `I = effect(A+B) - effect(A) - effect(B)`. Positive `I` with small single-site effects indicates backup or overdetermination. Candidates: mouth tokens and eye tokens; ViT depth 8 and 16 and 24 in Qwen; all read layers early and late. Grouped knockouts over 8 regions, reusing the Shapley design in token space, gives all pairwise `I` values in one go.
- Hook sanity: self-patching must give exactly zero change. A test that cuts all image paths must reproduce the image-free output to numerical tolerance. Both are simple gates that catch most hook bugs.
- Corruption: for faces, a Semantic Image Pair is the same actor, same head pose, different AU. Generated pairs do this better than NOTICE's cross-emotion FER pairs, which differ in identity. Mean ablation should use a face-video token mean, not ImageNet.
- Vision tower claims: stimulus edit plus ViT-level patches give cue to vision token. LM patches give vision token to residual stream. Attention knockouts give residual stream to answer. Each link needs its own pair of necessity and sufficiency tests.
- LEACE use: set Z to the AU delta, fit on within-pair differences so identity is differenced out, fit on training actors only, apply at image positions, and measure answer change. Linear only, so report a nonlinear decoder after erasure. A surviving decoder means erasure was incomplete, and the null is then not interpretable. Fitting on natural videos instead would remove anything correlated with the AU, so emotion and actor would go too.
- CAVs: use only with paired counterfactuals, as a direction source. A natural-image CAV inherits correlations. Treat the TCAV score as a hypothesis.
- DAS: a good fit for "is there a subspace of mouth tokens which carries AU12". Train on pairs, evaluate held-out actors with IIA, and add the Grant et al. latent loss or report EMD. Cost is unknown; run on cached activations to avoid repeated vision forwards.
- Causal scrubbing: fits the region hypothesis. If "only tokens at mouth and brow matter" is true, resampling all other image tokens from other same-label items should preserve behaviour. Resample within actor and emotion so distribution is preserved.
- SAE: skip in the 4 to 8 week plan. 243 items at 280 tokens is about 68,000 token vectors per layer (computed), small relative to what SAEs usually train on (not sourced), and the negative results make them poor value. VISTA retrains the projector, so it changes the model under test and is excluded.
- Steering: skip as a primary test. Use only as a sufficiency probe after a mediator is identified.
- FCCT's last-token finding is compatible with the audit. Its prompt names the object in a yes-or-no question and has no competing options, and the last token can be a true readout. In the option-list design it is contaminated. So the right response is to change the prompt and intervene on attention, not to abandon the last token.

### Gaps
- No located paper does cue-specific mediation through image tokens to answer on faces or video.
- No study of Qwen3-VL DeepStack, Gemma per-layer embeddings or Molmo2 under causal tracing.
- The original author list for causal scrubbing as "Chan" was not confirmed in results. Search results show Goldowsky-Dill, Radhakrishnan and others. I recall an arXiv paper by Chan et al.; not located.
- Attribution of activation patching to Cammarata not located. Secondary sources attribute causal tracing, interchange and path patching as above.
- [Rushing and Nanda](https://arxiv.org/abs/2402.15390) and [Mueller](https://arxiv.org/abs/2407.04690) full experiments not read.
- Gemma 4 attention mode per layer in E4B, and exact order of image and text in each processor, not checked in full.
- Basu et al. and Cao et al. were seen only through the blog.
- No DAS or pyvene application on a VLM located. pyvene handling of Molmo2 remote code unverified. [nnsight, MIT](https://github.com/ndif-team/nnsight) is an alternative; plain PyTorch hooks are probably simplest given the project's existing scripts.

## 5. Causal inference statistics

### Takeaway
Use within-item counterfactual contrasts on a continuous logit outcome, with actor as the resampling unit. Replace Holm on per-layer permutation p-values with step-down maxT, whose resolution floor does not grow with family size. Flip rate should be secondary, reported against a measured nuisance rate and an equivalence bound.

### Cited Findings
- [Phipson and Smyth 2010](https://arxiv.org/pdf/1603.05766): randomly drawn permutation p-values should be `(b+1)/(m+1)`-style, never zero; software is commonly understated by about 1/m.
- Westfall-Young maxT: the [adjusted p compares each statistic to the maximum across hypotheses under the group; Monte Carlo sampling still controls FWER](https://arxiv.org/pdf/2307.12832). [The R package flip implements step-down maxT](https://rdrr.io/cran/flip/man/npc.html).
- Sign-flip invariance [is equivalent to symmetry of marginals about zero](https://arxiv.org/pdf/2307.12832).
- GPD tail: [Knijnenburg et al. 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2687965) extrapolate small p-values from limited permutations. A [2026 preprint, permApprox](https://arxiv.org/html/2602.22975v1), reports GPD can yield zero when the fitted tail is bounded and proposes a support constraint.
- Stimuli as random: [Judd, Westfall, Kenny 2012](https://www.r-bloggers.com/2012/06/the-stimuli-as-a-fixed-effect-fallacy/) (secondary); [Westfall, Nichols and Yarkoni](https://doaj.org/article/a0857cda0d534f1ebe2586d876275042) on the stimulus-as-fixed-effect fallacy. [Type 1 error can exceed 50 percent under realistic conditions](https://link.springer.com/content/pdf/10.3758/s13428-016-0779-0.pdf) (attribution from snippet).
- Equivalence: [TOST against a smallest effect size of interest](https://www.scinapse.io/papers/2788930055); [TOSTER](https://psi-research-site.pages.dev/paper/lakens_2017_equivalence). Critique: [setting bounds by detectable effect is circular](https://www.r-bloggers.com/2016/05/absence-of-evidence-is-not-evidence-of-absence-testing-for-equivalence/) (attribution from snippet).
- Metric: [logit difference preferred to probability](https://arxiv.org/html/2309.16042).
- Preregistration: [Lakens, When and How to Deviate](https://doi.org/10.31234/osf.io/ha29k) (title only seen). [A 2024 AMPPS paper proposes a Preregistration Deviations Table](https://journals.sagepub.com/doi/10.1177/25152459231213802). [Rubin 2024](https://arxiv.org/pdf/2408.12347v2) argues preregistration does not improve severity evaluation where deviations are allowed.
- Project state: official gate Holm floors 0.036, 0.042, 0.032, all above 0.01, Branch B locked, "do not retune" (project file `research/STATE.md`). Pilot used 1,000 permutations. Grouped T1 is Leave-One-Actor-Out (project file `research/DECISIONS.md`).

### Inferences

Resolution (computed):
- Floors match exactly: `k / 1001` for k = 36, 42, 32 layers gives .036, .042, .032. The gate could not be passed at α = .01 at any effect size. It measured permutation count, not signal.
- Holm at α needs `m+1 > k/α`. For k = 42 and α = .01, m is at least 4,199. Use 50,000 for margin, or maxT.
- maxT adjusted-p floor is `1/(m+1)` regardless of k, and it exploits layer correlation. With 1,000 permutations the floor would be .001.
- Sign-flip by cluster: with n actors the exact two-sided floor is `2/2^n`: 6 actors .031, 8 .0078, 10 .0020, 12 .00049. Check the number of actors in any subset; with 6 it fails α = .01 for any effect.

Governance: keep Branch B locked. The notes in `research/STATE.md` already say not to retune. Finding the floor is a design flaw to report, and any improved test goes in a new prereg for the new study, not a rescue of the old gate. Use the deviations table. Pre-specify that the old gate is uninformative because of a resolution floor and show this computation, as a labelled post hoc diagnosis.

Pilot reading (computed): 5/180 is 2.8 percent, 95 percent Clopper-Pearson .9 to 6.4 percent, treating trials as independent. They were 180 flips counted on n = 36 items, so the true interval is wider. If 180 means 36 items by 5 conditions, the design effect could be meaningful. Rule of thumb: effective n = N / (1 + (m-1) ICC). With m = 5 and ICC = .3, effective n is about 82.

Flip-rate power (computed; exact one-sided binomial, 80 percent power, independent items, multiply by the design effect for clustering):

| Null rate p0 | True rate p1 | n at α = .05 | n at α = .005 |
|---|---|---|---|
| .028 | .10 | 70 | 125 |
| .028 | .07 | 165 | 290 |
| .01 | .05 | 110 | 180 |
| .05 | .10 | 170 | 300 |
| .05 | .15 | 60 | 100 |

Equivalence on flip rate (computed; n to show rate below margin, α = .05, 80 percent power):

| True rate | Margin | n |
|---|---|---|
| .03 | .10 | 90 |
| .03 | .08 | 130 |
| .01 | .05 | 125 |
| .05 | .10 | 180 |

Design effects (computed): ICC .1, 4 per item 1.3; ICC .2, 4 per item 1.6; ICC .3, 6 per item 2.5.

Recommendations:
- Primary outcome: within-item change in logit difference (target minus competitor), or log-odds of a pre-specified label. Flips discard magnitude, and they are near a floor because a base image is often far from a boundary. A flip is secondary.
- Measure a nuisance distribution rather than assume zero: patch from a same-class, same-AU clip, or from a re-encoded version of the same clip. The null rate p0 above should be this.
- Model: `logit_diff_delta ~ dose * cue + (1 + dose | actor) + (1 | item)` per model; do not pool models. Inference by actor-level sign flip or cluster bootstrap. For binary flips, logistic mixed models may separate with rare events, so use penalised or Bayesian priors, or simply cluster bootstrap.
- AIPW: in within-item randomised interventions, the experimenter sets treatment, so there is no propensity to model and adjustment is for precision only. It has value only if you later analyse natural AU variation, where actor and emotion confound. There, cross-fit it, and label the result observational.
- Multiplicity: use a hierarchical structure, with one family per planned claim. Gate-keep: stimulus effect first, then vision token, then path. Within a family, use maxT. Screening heat maps stay exploratory and confirm in held-out actors.
- Split actors into discovery and confirmation before any heat map.
- Equivalence margin: derive it from the human effect size or from the full-image-swap effect (for example, 10 percent of it), not from what is detectable.
- Stimulus generalisation: each actor and clip is a sample. With few actors, claims stop at this stimulus set.

### Gaps
- Original Judd et al. and Yarkoni texts not read. Lakens criteria not read.
- No source on AIPW in intervention designs; the advice above is reasoning.
- No published ICC for flip outcomes in patching studies. No published VLM flip-rate power study.
- GPD inside step-down maxT was not found in a source.
- The number of actors in the 36-item pilot was not checked.

## 6. Controlling text confounds

### Takeaway
Remove options from the prompt, read labels from logprobs at a fixed prefill, and take vision measures at image-token positions, where causal masking should make activations independent of later text. Test that independence as a unit test. Option lists stay only as a Latin-square control arm.

### Cited Findings
- Selection bias: [LLMs favour certain option labels](https://arxiv.org/pdf/2309.03882); removing option IDs reduces it, and shuffling IDs barely changes the standard deviation of recall, per [a separate study](https://arxiv.org/html/2409.18857v2) (attribution from snippets, verify). A [performance gap of 13 to 75 percent on reordering](https://arxiv.org/abs/2308.11483v1) is reported in one paper, attributed by snippet.
- Mitigations: [circular option shift for vision-language models](https://arxiv.org/pdf/2401.14011); [batch shuffle](https://arxiv.org/pdf/2409.14175); [SCOPE counterbiased placement](https://arxiv.org/html/2507.18182v2).
- Not universal: [clinical MCQ found no meaningful effect of position randomisation](https://arxiv.org/pdf/2506.00612).
- Neo et al. used [free-form and binary prompts with prefilled answers such as "It is a"](https://arxiv.org/html/2410.07149v2), not MCQ. NOTICE [balanced answer position](https://arxiv.org/html/2406.16320v3).
- Project audit (from brief): last-token activations decode emotion at about 100 percent from the option list. Project primary metric is free-response judge accuracy, 4AFC secondary (project file `PIVOT.md`).
- Patching site choice matters: [token choice changes findings](https://arxiv.org/html/2309.16042), and NOTICE [patched "Assistant:" for LLaVA](https://arxiv.org/html/2406.16320v3).

### Inferences
Prompt arms, in order of preference:
1. **Blind prefill**: fixed text with no labels, for example "Describe what this person is feeling. They feel" then read a logprob vector for a pre-registered label lexicon, including synonyms and plural forms. Last token and prompt are the same across items, so only image content can drive item differences.
2. **Single-probe**: "Is this person feeling X?" with yes and no logprobs, for each X in a Latin square. Good for dose-response and comparable with human yes-or-no tasks. Watch acquiescence bias, measured by an image-free baseline.
3. **MCQ with shuffle**: only to bridge to existing 6AFC results. Latin square over option order, label letters and label set. Score by logprob over option letters, averaged over rotation.
4. **Free response**: sampled for validity on a subset, judged by the existing synonym map. Too noisy and costly as a patching outcome. Use KL or logit difference at the first answer token as the mechanistic metric; [KL is reasonable for circuit discovery](https://arxiv.org/html/2309.16042).

Baselines:
- No image, same prompt: reports prior over labels; subtract this. Note how the model's processor handles missing image tokens, since sequence shifts.
- Phase-scrambled or tile-shuffled image, and a different actor's face: tests face-specific information beyond low-level image statistics.
- Cross-frame shuffle for dynamics; first frame only; peak frame only.
- Each gives a floor for Recovery Rate.

Separating vision from text:
- In a causal LM, tokens placed before the question cannot see it. So image-token activations should be identical across prompts if the image comes first. Unit test: change only the option list and assert `||a_img(P1) - a_img(P2)|| < eps`. Any leak means a system prompt or a bidirectional mask is involved; Gemma has bidirectional handling in some attention modes, and that needs checking.
- This invalidates nothing about probes, but it removes the option-list route, so an image-token probe can only decode what the image holds. It still may decode actor, background or lighting. Use grouped actor cross-validation and within-pair differences.
- Decompose the last token: fit a per-item variance split into prompt, image, and interaction components across the Latin square. Report image-attributable variance per layer, which directly gives the "how much of the last token is vision" quantity.
- Label-text cross-talk: in single-probe, the prompt holds X, so a probe for X at the last token is trivially text. Decode only the answer or a contrast between images under the same X.

### Gaps
- No located emotion-recognition VLM paper on option-order effects.
- Option-order numbers come from snippets, and the mapping of figures to papers needs checking.
- Whether each model's chat template places images before all text in this project's pipeline was not checked in `scripts/`.

## 7. Human-model alignment for the same intervention

### Takeaway
The strongest comparison is same stimulus, same edit, same response format, with dose-response curves compared on shape and threshold, and all alignment reported relative to a human split-half ceiling. New human data on edited stimuli cannot be avoided for this claim, and it will not be free on Prolific; use a departmental pool or restrict the human claim.

### Cited Findings
- [Error consistency](https://arxiv.org/pdf/2006.16736.pdf) tests whether two systems fail on the same individual inputs. CNN-human consistency was little above chance while CNN-CNN was high.
- [Alvarez et al. 2024](https://konklab.fas.harvard.edu/Papers/Alvarez_2024_UniReps.pdf) argue that internal human noise means comparisons need correction. Decision-margin consistency correlates per-item margins and found the human-machine gap larger than estimated.
- [Klein et al. 2025](https://par.nsf.gov//servlets/purl/10654568) treat uncertainty in error consistency (from search summary, not read).
- Cue conflict: [Geirhos et al.](https://arxiv.org/abs/1811.12231v3) used images with conflicting cues to expose texture bias in CNNs, a paradigm that transfers directly to AU conflicts.
- Human paradigm: [Generative Face Grammar reverse correlation](https://pmc.ncbi.nlm.nih.gov/articles/PMC4927208) links random movements to labels.
- Cost: [Prolific charges academics a 33.3 percent fee, minimum £6 per hour, recommended £9](https://www.prolific.com/pricing). No free tier for data collection.
- Tools: [psignifit repo exists, licence non-standard and not read](https://github.com/wichmann-lab/python-psignifit).

### Inferences
Alignment criteria, weakest to strongest:
1. Same direction of effect for each cue.
2. Same rank order of cue importance (Shapley region ranking).
3. Psychometric match: fit `P(label | dose)` per system. Compare point of subjective equality and slope with CIs. Models are deterministic and may have compressive logits, so compare after a monotone link or at the rank level.
4. Item-level: decision-margin consistency corrected by human split-half reliability, from participant random split.
5. Interaction: same cue-by-emotion and cue-by-time pattern.
6. Cue-conflict: for conflicting brow and mouth, who wins per item.
Report each against the noise ceiling (human split-half with Spearman-Brown). A model cannot be said to beat the ceiling, and the comparison should be across the same items.

Do not conclude "same mechanism" from same curves. Matching curves can come from different internal routes; that is why internal mediation is a separate claim.

A model has no trial noise and answers each item once. Comparing one deterministic model to an average human ignores individual differences. Use human-to-group fit as the comparison distribution for model-to-group fit.

Costing: 60 participants, 20 minutes, at £9 per hour is £180 plus 33.3 percent fee, £240 (computed). Under the no-cash rule this is excluded unless already funded. Options: the Cambridge departmental participant pool (not verified), a collaborator's pool, or existing EU-Emotion norms for unedited clips only. Without new human data, the model results still stand as a causal statement about the model, and the human comparison is limited to diagnostics from the literature (for example mouth for happiness). State that limit in the paper.

Human stimuli need the same edit validation: a short realism and emotion rating of edited and null-edited clips. If humans notice edits, the same edit is not the same stimulus. Ethics approval is needed for any new data, and edited faces of consented actors need a licence check against the EU-Emotion terms, which I did not read.

### Gaps
- Noise-ceiling method sources for this exact case were not located separately. Alvarez et al. is closest.
- Klein et al. not read; Calder composite-face emotion paper not found in search.
- Cambridge participant pool cost, and any free online hosting (jsPsych, cognition.run), not verified.
- Whether EU-Emotion's licence allows edits or redistribution is unknown.

## 8. Recommended stack, 4 to 8 week plan, and failure modes

### Takeaway
Build the prompt fix and hook gates first, as they are cheap and could invalidate everything else. Then run a two-arm stimulus experiment (FACSHuman and LivePortrait with null-edit control) with blind prefill and logit outcomes, followed by top-down complete cuts through vision and attention paths. A publishable minimum is a cue-level necessity and sufficiency map plus coarse mediation in Qwen3-VL-8B by week 6, extended to the others by week 8.

### Cited Findings
Stack, all located above:
- Stimulus: [FACSHuman](https://github.com/montybot/FACSHuman) (LGPL), [LivePortrait](https://github.com/KlingAIResearch/LivePortrait) (MIT plus InsightFace research), [MagicFace](https://github.com/weimengting/MagicFace) (Apache) for single-frame.
- Verification: [LibreFace](https://github.com/ihp-lab/LibreFace), [OpenFace 3.0](https://github.com/CMU-MultiComp-Lab/OpenFace-3.0), [ME-GraphAU](https://github.com/CVI-SZU/ME-GraphAU), ArcFace via InsightFace, research-use weights.
- Intervention: [nnsight](https://github.com/ndif-team/nnsight) or plain hooks, [pyvene](https://github.com/stanfordnlp/pyvene) for DAS, [concept-erasure](https://github.com/EleutherAI/concept-erasure).
- Not chosen: [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) and [SAELens](https://github.com/decoderesearch/SAELens) are active and MIT, but I found no evidence that they support these VLMs.
- Stats: [flip](https://rdrr.io/cran/flip/man/npc.html), [TOSTER](https://psi-research-site.pages.dev/paper/lakens_2017_equivalence), own sign-flip code.
- Project budget estimates: patching 8 to 24 GPU-hours for Qwen, all-layer extract 4 to 12 (project file `research/COMPUTE.md`, planning figures).

### Inferences

**Weeks 1 to 2: gates (no new stimuli).**
- G1 Timing: one smoke job per model. Set the assumed 1 to 3 s forward.
- G2 Image-token invariance across option lists.
- G3 Hook tests: self-patch gives zero. All-vision cut gives image-free output, with Qwen `inputs_embeds` plus DeepStack, and Gemma with per-layer input.
- G4 Positive control: full image swap between clips with different labels, at token level. Dose: 10, 25, 50, 100 percent of tokens. Expect a large monotonic effect. If not, the model is not image-driven on this prompt, and the study stops or changes prompt.
- G5 Prompt arms: blind prefill accuracy against existing 6AFC.
- G6 Variance split on last token across Latin square.
- About 15 to 30 GPU-hours across models (estimate from project figures).

**Weeks 2 to 4: stimuli.**
- Choose cues: AU12, AU4, gaze, head yaw, dynamics. Pick 40 clips balanced by actor, with discovery and confirmation actors split.
- FACSHuman: render AU ramps on several identities.
- LivePortrait: learn AU-to-keypoint map, add null-edit and dose levels. At the 4090 rate, 182,000 frames would take about 40 minutes (computed; 150 frames, 243 clips, 5 levels), with I/O and face detection extra. On composite grids it is far fewer.
- Pass manipulation checks. Log exclusions. Register before the VLM sees edited stimuli.

**Weeks 4 to 6: stimulus tests, then vision.**
- Dose-response of blind prefill logits by cue.
- Region Shapley over 8 regions in pixel space, 40 items, Qwen first (about 3 to 9 GPU-hours per model, computed).
- Token-level: replace region tokens, with buffer sizes 0, 1, 2 (per Neo's result on buffers).
- Necessity and sufficiency patches at cut-points in the table in section 4.

**Weeks 6 to 8: path, redundancy, and second model.**
- Attention knockout from all text to image tokens, by bisected layer windows, with sufficiency reverse.
- Joint region and joint layer interaction contrasts.
- LEACE at image positions, fit on within-pair differences. DAS if time remains.
- Repeat on Gemma, then Molmo, adapting cut-points.
- Human arm only if funded or a free pool is confirmed. Otherwise write as model-only causal study.

**If only 4 weeks:** gates, FACSHuman plus LivePortrait for AU12 and AU4 only, Qwen only, stimulus dose-response and all-token plus region-token necessity and sufficiency. Drop Shapley, LEACE, DAS, Gemma and Molmo.

**Decision rule for null results.** A null counts only if: G3 and G4 passed; dose was shown effective elsewhere; joint ablation was also null; equivalence passed against a pre-set margin; and divergence was reported. Otherwise report "inconclusive".

**Failure modes:**

Stimulus:
1. Generator artifact read as a cue. Mitigate by null edit, cross-generator replication.
2. Null-edit too good or too bad: a null edit that changes the model's answer means the base rate has shifted. Report that effect first.
3. AU edit changes identity, age, or perceived gender.
4. Off-target AUs: AU12 raising cheek AU6, or head pose coupled to the edit.
5. Detector fooled by the generator, so that the manipulation check passes falsely.
6. Detector poor for the AU (AU15, AU20), so check fails falsely.
7. Synthetic arm not processed as a face or as a person: VLM may answer "cartoon" or "3D model".
8. Temporal flicker from per-frame editors becomes a dynamic cue.
9. Compression and resize wipe out subtle AUs before tokenisation. Gemma 48 px per soft token; Qwen 32 px (16 by 2); on a 4 by 4 grid with 280 tokens, about 17 tokens per frame or roughly 4 by 4 per face (computed). A lip corner may be sub-token. Cues below this cannot be tested, and a null here is about resolution, not representation. Test native single-frame input too.
10. Grid layout confound: composite grids add spatial position and frame order, and dynamics in a grid are not motion. A temporal claim needs native video or frame-order shuffle.
11. Qwen's temporal patch merges 2 frames per token, so per-frame token analysis is not available on video input.
12. Licences: InsightFace weights, FLAME, and EU-Emotion actors' consent. Edited faces should not be redistributed by default.
13. Human-perceived fakeness alters emotion, so the same edit is not perceptually the same.
14. Parametric avatars lack skin-folding detail, so wrinkle-based AUs such as AU4 and AU6 may be underrepresented.

Prompt and scoring:
15. Blind prefill has low accuracy, so there is little signal to mediate.
16. Lexicon bias: labels have different priors and tokenisation. Multi-token labels need joint logprob.
17. Prefill changes the task from recognition to completion.
18. Gemma thinking mode or chat template shifts the answer position.
19. Latin-square arms inflate job count.
20. Acquiescence bias in yes-or-no prompts.
21. Fine-tuned checkpoints differ; keep the baseline only.

Internal:
22. Text leak in activations through system prompts.
23. Patch misses DeepStack or per-layer inputs, giving a false null.
24. Hook mismatch across Transformers versions. The project pins Molmo to 4.57.3 and others to 5.10.1 (project file); pin commit SHAs.
25. Image-position mask misaligned for video or merged tokens, giving silent wrong patches.
26. Spatial token map wrong (Where To Look warns of [ViT mapping limits, and that SigLIP was untested](https://arxiv.org/html/2608.10758v1)); Molmo pooling and Gemma pooling need own maps.
27. Information leaks beyond region tokens, through attention in the ViT. Region-token patching post-ViT does not remove what neighbours carry. [Neo's buffer effect](https://arxiv.org/html/2410.07149v2) and the low correlation in Where To Look point this way.
28. Ablation by mean is off-distribution, particularly for image tokens with [larger norms than text](https://arxiv.org/html/2410.07149v2).
29. Self-repair and LayerNorm rescale after zero ablation.
30. Backup paths and overdetermination, so a single-site null. Joint tests.
31. Hidden pathway: a patch effect arising only from an off-manifold state. Report divergence.
32. Non-transitivity: cue changes vision tokens, tokens change answer, yet cue does not change answer via them.
33. INT large, so one-direction patching misleads.
34. Dormant effects in other contexts: test across prompts and emotions.
35. Heat-map forking paths: screening then confirming in the same actors.
36. Recovery Rate unstable when clean minus corrupt is small; [values fall outside 0 to 1](https://arxiv.org/html/2608.10758v1). Exclude by pre-set threshold on total effect.
37. Erasure removes correlated concepts, or leaves nonlinear residual.
38. CAVs and probes confounded by actor and lighting.
39. DAS overfits small n; held-out actors required.
40. Causal scrubbing depends on the resampling distribution.
41. Model-specific findings: three architectures need three sets of cut-points, so pooling is invalid.
42. Mechanistic conclusions from object-based literature may not transfer to affect.

Statistics and inference:
43. Flip floor: base images far from boundary, so effects are invisible. Use logits.
44. Too few actors for exact sign-flip resolution.
45. Clustered items treated as independent.
46. Repeating the Holm floor by re-using `m = 1000` for a larger family.
47. Equivalence margin chosen post hoc.
48. Rescuing the locked gate by retuning.
49. Null reported as evidence of no use, without gates.
50. Stimulus sample too small to generalise across faces, ethnicity, age and setting. Actor set may have demographic skew.
51. Human-model match over-interpreted as same mechanism.
52. Multiple testing across three models, cues and paths; hierarchy needed.

Operational:
53. 12 h walls: checkpoint per item, with resumable arrays.
54. CSD3 queue or SSH outage; keep caches and sync plan.
55. Storage: per-layer per-token activations are large. 243 items, 280 tokens, 42 layers, 2,560 dims, float16 is about 15 GB per condition (computed), before multiplying by edits, doses and frames. Cache only cut-point layers, and delete redundant caches.
56. Remote-code model updates break hooks.
57. Wall-clock overrun: LivePortrait AU mapping and AU verification can eat two weeks. Fall back to driving-clip transfer (neutral versus actor's own target frame) as the edit.

### Gaps
- All GPU-hour figures depend on the unmeasured forward-pass cost.
- Whether LivePortrait AU-mapping can meet a selective AU edit is unknown until tried; no published evidence. Community slider tools were not reviewed.
- Whether any of the three models shows enough image-driven emotion signal under a blind prompt is unknown. This is the biggest risk, and gate G4 and G5 resolve it first.
