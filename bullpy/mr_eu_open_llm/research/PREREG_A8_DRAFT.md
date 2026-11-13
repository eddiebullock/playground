# DRAFT amendment A8: larger open-weight models through the frozen pipelines

**Status: FROZEN 2026-11-09 (design text), tag `a8-freeze`, before the Qwen3-VL-8B A1 result was read and before any A8 model ran on any study stimulus. Code pins follow as a dated pre-data step (Freeze checklist item 2); later changes are dated deviations. The filename keeps "DRAFT", as the other amendments do, so that references in code stay valid.**

Current `design_sha256` of the block below: `95dd41b5d3c33b79ee9e54a20d34ea1e18b6c9156fcabd719ebfb084709b7b50`

The hash is the sha256 of the bytes between the fence lines of "Design block", without the fences, computed by `research/scripts/a8_design.py` in the same way `scripts/a7_design.py` computes A7's. Any edit to the block changes it, and the value pinned in `a8_design.py` must change with it, on purpose. Earlier block hashes: `49665a159771043e50e4799d972e3deaa84b1f03a04009bdb458abf6fbea11eb` (before the audit) and `f14fdcb7c0d24ca9d5443e362d68947cf52e43610b90f06c970088492e6b764c` (after the audit, before the re-check); the changes are in "Audit rulings".

**Written:** 2026-11-09, while the Qwen3-VL-8B A1 arrays (39982137, 39982138) were still running and before any A1 rotation file was read. Lead decisions and an independent audit (`research/reviews/a8_audit.md`, 21 findings) were applied the same day. Supporting checks: `research/notes_a8_models.md`.

**What was known when this was drafted.** The A3 result for the three current models (Qwen3-VL-8B accuracy .282 on `single_face`, Gemma 4 E4B .371, Molmo2 .072; `RESULTS_REGISTRY.md`, A3 section), the vision gates (Gemma and Qwen pass G2 to G4, Molmo fails), input delivery, the readout ladder, and A2, A4, A5 and A7 as frozen. No A1 result. The motivation is that the two eligible models are small and weak on the task, so confirmatory claims rest on two weak models. The A8 models were chosen on licence, size, family and storage, before either produced an output on any study stimulus.

**Order of freezing.** The design text and its block are frozen before the Qwen3-VL-8B A1 result is read. Code pins ("Code hashes") follow as a dated, pre-data step, before any A8 output exists; no scored A8 entry point runs until they exist. Shard counts and GPU counts are run settings (see "Run settings"), set by dated notes after the timing smokes.

**Depends on:** the frozen A1 (`research/PREREGISTRATION.md`, "Amendment A1" and "A1.1"), A2 (`research/PREREG_A2_DRAFT.md`), A3 (`research/PREREG_A3_DRAFT.md`) and A4 (`research/PREREG_A4_DRAFT.md`), unchanged, and the DECISIONS entries cited below.

**What A8 is not.** It does not change any frozen threshold, family, estimand, prompt, readout, item set, code file or verdict. It cannot rescue a frozen verdict and cannot weaken one. Its results are reported beside the frozen ones, in their own rows, under their own multiplicity control.

## Lead decisions, 2026-11-09

Before any A8 output and before any A1 result was read:

1. Qwen3-VL-32B is confirmed. Gemma 4 31B was conditional on storage on the lab project store; the user approved that storage on 2026-11-09, so Gemma 4 31B is admitted inside the block (`"status": "admitted_2026-11-09_user_approved_storage"`), and the dated note below is the record.
2. Two models only. InternVL3.5 and Mistral-Small-3.2 are future work.
3. A1 on the 32B models runs through the sharded runner, every shard computing mismatch partners over the full item set, checked by the equivalence tests in the block. `ampere-long` is not open to this account (Compute); it is a fallback only if access is granted.
4. A8 jobs queue behind the frozen Phase 2 runs, which must be COMPLETED (audit finding 3).
5. Gemma 4 31B gates are strict: if the frozen `gemma4` gate adapter fails on 31B, there is no gate result and the model is ineligible for A2 and A4.
6. Natural-clip RQ1 runs on A8 models, descriptive only.
7. The design text is frozen before the Qwen3-VL-8B A1 result is read; code pins follow, dated and pre-data.
8. Budget is phased with the stop rules below. Up to about 600 GPU-hours for Qwen3-VL-32B is accepted (free allocation).

## Question

Do the frozen tests give the same answers on larger, newer open-weight models in the same families as the eligible models? A8 asks this of RQ0 (A1 and A3), the vision gates and delivery, Gate B and the stimulus arm (A2), the randomised cue arm (A4), and, descriptively, natural-clip RQ1. Each is judged by its frozen rule.

## Models

Free, open weights, Apache-2.0, not gated, verified on Hugging Face on 2026-11-09 (API `sha` of `main`; details in the notes file).

| A8 id | Status in the block | Hugging Face repo | Pinned revision | Params | bf16 weights | Frozen code key | A1 sampling |
|---|---|---|---|---|---|---|---|
| `qwen3vl_32b` | confirmed | `Qwen/Qwen3-VL-32B-Instruct` | `0cfaf48183f594c314753d30a4c4974bc75f3ccb` | 33.36 B | 66.7 GB | `qwen3vl` | sequential |
| `gemma4_31b` | admitted 2026-11-09, user approved storage | `google/gemma-4-31B-it` | `842da3794eaa0b77d5f08bae87a17459d91ff475` | 31.27 B | 62.5 GB | `gemma4` | batched |

Both are Apache-2.0 (Gemma 4's card `license_link` serves the Apache 2.0 text). Sampling follows the frozen rotation script's rule by key.

**Why these.** Each frozen pipeline script whitelists the keys `qwen3vl`, `gemma4` and `molmo2` and branches on them. Qwen3-VL-32B has the class (`Qwen3VLForConditionalGeneration`) and vision tower of Qwen3-VL-8B; Gemma 4 31B has the class (`Gemma4ForConditionalGeneration`) of Gemma 4 E4B. Both load through `scripts/evaluate.resolve_model_path`, which reads `MR_EU_HPC_MODELS_DIR`, so they run under the existing keys with a different weights directory and different output paths, and no code edit. Each is a scale-up of an eligible model. Gemma 4 31B also differs from E4B in more than size (no KV sharing, no per-layer inputs, no audio tower).

**Future work, not in A8.** A third family (InternVL3.5-38B-HF, revision `7c830fc2...`, whose language model is Qwen3; Mistral-Small-3.2-24B-Instruct-2506, revision `95a6d26c...`, which ships no video preprocessor config). Either needs new model code in every pipeline and its own amendment.

**Not proposed.** Gemma 3 27B (Gemma Terms of Use, gated), Gemma 4 26B-A4B (mixture of experts), Qwen3.5-27B (no key in the code), Cosmos-Reason2-32B (a task-tuned fine-tune), Llama 4 Scout (gated, too large for one A100), larger Molmo2 checkpoints (Molmo2 fails G2 and G4).

## Storage

**Constraint.** RDS hpc-work is at 1041.9 GB of a 1099.5 GB soft limit (1209.5 hard) and 996,131 of 1,048,576 files (`quota`, 2026-11-09). The stop rule in the block (1,080 GB, 1,040,000 files) is checked by the launcher before every job. No 30 B class model fits as a new download on RDS. `/home` is at 49.8 of 52.4 GB. Slurm reports `TMP_DISK` 0 on ampere nodes.

**Qwen3-VL-32B: read in place.** `/rds/project/rds-ePtR33Nsgi4/eb2007/models/Qwen3-VL-32B-Instruct`, in the user's own directory of the lab project store (own quota: 14,341 of 23,000 GB; 7.70 M of 11.78 M files), owned by `eb2007`. Its download metadata name revision `0cfaf481...`, its shard sizes match Hugging Face, and the stored sha256 of shards 1 and 14 match the Hugging Face LFS hashes. CASCADE loads this copy (STATE.md), so A8 only reads it. A8 needs one symlink, `~/rds/hpc-work/study3/a8_models/qwen3vl -> <that directory>`.

**Gemma 4 31B.** One download of the pinned revision into `/rds/project/rds-ePtR33Nsgi4/eb2007/models/gemma-4-31B-it` (12 files, 62.5 GB), with the Hugging Face cache in the project store, then the symlink `~/rds/hpc-work/study3/a8_models_gemma/gemma4`.

**Integrity.** Every weight file's name, size and Hugging Face LFS sha256 at the pinned revision is in the block. Before the first A8 job the coordinator runs `research/scripts/a8_preflight.py --full_hash` read-only on each copy and records the result; a differing sha256 stops A8 for that model. Every job then runs the same checks without the full hash (names, sizes, resolved path), and a full hash runs again after the last A8 job.

**Outputs.** Cells stay JSON lines. All A8 output goes under `results/a8/` and `research/reproduction/a8/`, with one declared exception: the frozen natural-clip script always writes `research/reproduction/same_clip_family_model_<model>.json`, so the launcher refuses if that file already exists and moves it into `research/reproduction/a8/<a8_id>/` after the run.

## Design block

```json
{
  "amendment": "A8",
  "models": {
    "qwen3vl_32b": {
      "status": "confirmed",
      "repo": "Qwen/Qwen3-VL-32B-Instruct",
      "revision": "0cfaf48183f594c314753d30a4c4974bc75f3ccb",
      "licence": "apache-2.0",
      "code_key": "qwen3vl",
      "weights_dir": "/rds/project/rds-ePtR33Nsgi4/eb2007/models/Qwen3-VL-32B-Instruct",
      "models_dir": "~/rds/hpc-work/study3/a8_models",
      "expected_config": {
        "architectures": [
          "Qwen3VLForConditionalGeneration"
        ],
        "model_type": "qwen3_vl",
        "text_num_hidden_layers": 64,
        "text_hidden_size": 5120
      },
      "a1_sampling": "sequential",
      "weight_files": {
        "model-00001-of-00014.safetensors": {
          "size": 4932328944,
          "sha256": "6b9dfbc930e505402ae9d7e5091a9d7d656cda5f34614f01cfe70bfb0cca27cb"
        },
        "model-00002-of-00014.safetensors": {
          "size": 4875990528,
          "sha256": "d8bb44b4ff303fe76fe9e894022fb3dc71b15a2e716592790fe0e3c3e60478fa"
        },
        "model-00003-of-00014.safetensors": {
          "size": 4875990552,
          "sha256": "54f22e8b3168f8dc962fac0d313607ebf52a12b433d4cf3098a0d82d9f042940"
        },
        "model-00004-of-00014.safetensors": {
          "size": 4875990584,
          "sha256": "ad09c74d3c13ee29b5d0d84548fd8a3424a651564eaccd519946c296e59c557f"
        },
        "model-00005-of-00014.safetensors": {
          "size": 4875990584,
          "sha256": "fc993c8a0e2a5b0570f383e1a95dc3a1281d1b224b6f3ee908f4827941e1dfc2"
        },
        "model-00006-of-00014.safetensors": {
          "size": 4875990584,
          "sha256": "82f05620d1f718a90c362b221d6a184ff1a0f53301d706882d3df49695fa1974"
        },
        "model-00007-of-00014.safetensors": {
          "size": 4875990584,
          "sha256": "fb91da8cb01ff4de3eef0eab1c3e769a734b3a1aafc61734068638a0d6c86934"
        },
        "model-00008-of-00014.safetensors": {
          "size": 4875990584,
          "sha256": "431ca56535c8781944ce3801f5eb61c45531e853ecc5846d936ebaf4761b764f"
        },
        "model-00009-of-00014.safetensors": {
          "size": 4875990584,
          "sha256": "3825e3f4302f4d2f7d76aa7430d2ce0864fde6b9e540a5806bf0d8e38e4d9f47"
        },
        "model-00010-of-00014.safetensors": {
          "size": 4875990584,
          "sha256": "aded5a4d1d5e22dbd8b6f79266b6eb88c840411b09527c53917a1419ace22e2f"
        },
        "model-00011-of-00014.safetensors": {
          "size": 4875990584,
          "sha256": "3820ffe8d8d6477f6fe8d614ef3c87abb264ee39accebf43a1507b970d80946f"
        },
        "model-00012-of-00014.safetensors": {
          "size": 4875990584,
          "sha256": "05ad2d08ce71963121c9b03f1d9ec5d7641052f4b23c6c12b80d71065eb8e98e"
        },
        "model-00013-of-00014.safetensors": {
          "size": 4875990584,
          "sha256": "b64f2289871261fdd1abbd3b78bcd66011b341de3dc8eeb2ed1a473ee7c8d95c"
        },
        "model-00014-of-00014.safetensors": {
          "size": 3270697008,
          "sha256": "e45b6c9998c77ee5a6577f9f47bc76416c1d4d387169e50c4c9d3134ea51b13b"
        }
      }
    },
    "gemma4_31b": {
      "status": "admitted_2026-11-09_user_approved_storage",
      "repo": "google/gemma-4-31B-it",
      "revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
      "licence": "apache-2.0",
      "code_key": "gemma4",
      "weights_dir": "/rds/project/rds-ePtR33Nsgi4/eb2007/models/gemma-4-31B-it",
      "models_dir": "~/rds/hpc-work/study3/a8_models_gemma",
      "expected_config": {
        "architectures": [
          "Gemma4ForConditionalGeneration"
        ],
        "model_type": "gemma4",
        "text_num_hidden_layers": 60,
        "text_hidden_size": 5376
      },
      "a1_sampling": "batched",
      "weight_files": {
        "model-00001-of-00002.safetensors": {
          "size": 49784788364,
          "sha256": "eeef8791537bc04f110967c513149e037d2a9ae97d49add7291ebfa62806bbfa"
        },
        "model-00002-of-00002.safetensors": {
          "size": 12761549884,
          "sha256": "018912220f559f7025d60333e0996183cd538aa77ad6f4988a89ce47be681f10"
        }
      }
    }
  },
  "output_roots": [
    "results/a8",
    "research/reproduction/a8"
  ],
  "storage_stop": {
    "rds_gb": 1080,
    "rds_files": 1040000
  },
  "dtype": "bfloat16",
  "frozen_code_sha256": {
    "config.py": "75d4a11128aad7d026cbc66331ed9cfd0c51acee1b64a986c2b2138cd1ba77e8",
    "research/scripts/a4_detector.py": "3fbd35dee68615a6fe1165df57a4d290c8097e8bf2a5ac714ee73ba15f302e42",
    "research/scripts/analyse_rotation.py": "98890c5bff65f13abeb0549b61fe414dc9fa30082b84bcb564671683ec12e24f",
    "research/scripts/analyse_rotation_run.py": "7f52f9e7b29f0afbbe76977ee3b3330f01c8f6ececf29fe14ddf9655c67d653e",
    "research/scripts/decoder_preflight.py": "ed5262d89562e2662f6aea3660edb4f1d9cb5306a08ffc68daddd4f669c20699",
    "research/scripts/extract_au_measures.py": "bc846844a3ea37fe39b80e128e0e0753b7680a2976e62180b24b296cabe00ab8",
    "research/scripts/human_au_effects.py": "c02a945ba578156cc4394e7d3a49761ad6e74495585034fbb059a4ce40199a2b",
    "research/scripts/render_au_faces.py": "6bdfa4e947b06aa7c2213dc31fbd297df4f8b97ca045a40905c6de4800c001d4",
    "research/scripts/same_clip_cue_analysis.py": "26dc0a48de85a720c89142e856ccde8b73363ab2815de96e0034d5b3da9a7109",
    "research/scripts/same_clip_family_analysis.py": "2e633c534bf1bf0fbfd6d133aff587003be30578ee09ecf800d4eedce5de493f",
    "scripts/a4_analyse.py": "30f8f529d128b555ccfbf4d31ac3de70d91f90b1f5bfd52360ae9f39305fe56d",
    "scripts/a4_design.py": "a0bf7bb1c69ec7af1f11fabb5a9f31a6c694a30b92e624f1e1c241f161d062a8",
    "scripts/a4_run.py": "f842e574bf5cdaebbe9057d0ecc0e311c900dd54ff0ca221d1f9baa3768a8dc6",
    "scripts/activation_forward.py": "695b1bd7dbb86c2d141621057dd7854a5f48703cb56d70403a4c0b0f5aedec4f",
    "scripts/analyse_stimulus_interventions.py": "aba4f533daad6210e87f7bb4e68fc4627805cb4dfd9d9308df734fada4edc8b1",
    "scripts/emotion_parse.py": "19a8f875da09bdb4b7ba679eb90ee72adbb37c31c8f2b4e74e2c9e4d445ba93d",
    "scripts/eu_audio_resolver.py": "994947c7ef3a9b08dca7b1e0c007618daaadf37384c833f8defc2255a751d84a",
    "scripts/eu_emotion_definitions.py": "091b0129cea984596cd8b8675bc2b69e7f294810e0b66847ddea4013077bee0d",
    "scripts/eu_emotion_synonyms.py": "9f27ed1c9704ebb1ee59a50ac2f0030d80064e8c1c147f6964862663b76323f8",
    "scripts/evaluate.py": "cc672f9438fe0b165dc2d620f5fad0e0c3a8845857a91e68871e0dad669a2481",
    "scripts/face_input_validity.py": "8a681a652d8e34065372d4f45bce650ba5276b5138e3aa64cf4310d65e93d818",
    "scripts/forced_choice_entropy.py": "1578a7242f7217aaab6c2bcdaef8ce82df0261c05a249960df3646a6d1f2799f",
    "scripts/frame_sampling.py": "11b1bb42f8565e66ff0be457bd76e4649410a8991cfe841280eba2444d90becb",
    "scripts/free_response_judge.py": "29803031809f90c98d5428a782e26e90edcada8b6f6c8e0dceb90efd460be4bb",
    "scripts/human_entropy.py": "e585bf2317a939aa5a494fdcf21aad7850d41a9be3e0166f2f721ec06206fc9c",
    "scripts/mindreading_audio_resolver.py": "a13abf00d8f5cf094c46a470a9ff15bde2d72cd0640afa6203dad110c69ee81c",
    "scripts/model_compat.py": "0d88a2d824c966b046d27c6201c5c585c361eb516f30ef51e2b54bfe782efd7c",
    "scripts/model_inference.py": "2ddb946f75db80de14299a77b0de682d1b16ce092185f64d3b3e2d7ae0e7fc7c",
    "scripts/multi_frame.py": "2cdc5c454f0fe9ecccbd167d7b145d29b7a9d602bbb7a6e5cf61c28cbe4e1b35",
    "scripts/prompts.py": "871272d4aa6c86ff809f7631a4db81aeeb41564ba7f0a31809ea08e6dcf64ba7",
    "scripts/readout_ladder.py": "738b2c5031e5f7f1c0337896b0e8e680b2f767d0f33c850588f743d6c5595c1c",
    "scripts/run_stimulus_interventions.py": "f7880c94e9671cc294cd4eaeece2108d7cd1d7a4c7e6f9e30a1c495964e6cd82",
    "scripts/semantic_entropy.py": "058f47fcc7e9532fa1661f866f929de6449024664aacbda953aaa0ab4a0ab62f",
    "scripts/statistics.py": "2d9d361fb9087e206d8896f343707109bb531ae4bdacd0cfab4049630e6ee7b7",
    "scripts/stimulus_design.py": "89d256c1072e0768fc747294553c5710adaf42184a175866d7bd75bcc6104978",
    "scripts/stimulus_interventions.py": "93c601eb42d6700d4f0fc8ba72317ed5c2e354105d57fcac611c59660125deb3",
    "scripts/tolerant_parse.py": "d4e31b4f94eb28480edb1f2a9e41ed12a6f83530dd5545881bd56d0fcfa58833",
    "scripts/trial_foils.py": "5e5d23a085ee4465d983e3bf12f7de48d95b0a2fd16287b6529bdf3c093d9ebf",
    "scripts/vision_gates.py": "6327b6f88e99409cba5c63d5aca0fe97348fbce4f3b08522158a8d9ba3ce8194",
    "slurm_jobs/study3_a4.sh": "dd9a360b20701742fd70d849b92c964d6b992c966ded744fa4125e0166103c26",
    "slurm_jobs/study3_face_input_validity.sh": "2bf3a9f74e6d6a2929e34fc54249e79ea9216b50460f03c89c1a7d3e058d0d8a",
    "slurm_jobs/study3_readout_ladder.sh": "a4ed37971dc28348f9d67052804b72a530c1cf59537d5116d60f7d04be9f39bd",
    "slurm_jobs/study3_rotation_eval.sh": "59fe95123cc83b5710bb47de824c608cf4f6fac8ebd864d383e77eefe71e6230",
    "slurm_jobs/study3_stimulus_interventions.sh": "1c8d22ea637abedd3a3c2332331639c7e33a7e7ed9cb2e741335d17ce138eec5",
    "slurm_jobs/study3_vision_gates.sh": "3607c0407319d9b4fdd333102e4776765e510470874cd243301ba6a11c97b705",
    "src/paper_v2/metrics.py": "1fc15676cae999e5185cfd016e3dd9db616c71ec0bb4c5d38cd49a4e7b886568",
    "src/paper_v2/option_score.py": "ae949e4c1c3abc6a775a535c4e0a5042bce59ba14f7ac1afeefd476f032a5a2b",
    "src/paper_v2/schema.py": "4acc626669439384115ae0efcb52cec4edee02c07662a242212d5f7b9c359b47"
  },
  "frozen_a3_summary_sha256": "c1c64e6975b32356a3314268c054f0dee483ddf74b11cd66d03b2faf972e64a4",
  "frozen_design_sha256": {
    "A2": "0ae22118d3b2690c581595890b9bb8b904e1f7860f37be9ac909f925b7058de7",
    "A4": "12c454da512bfe4c239e6c3ef3610d80df6fd7c2e9119a9ff8df913a3c1a3e94"
  },
  "steps": [
    {
      "step": "vision_gates",
      "source": "DECISIONS 2026-11-04 and 2026-11-05; scripts/vision_gates.py GATE_RULES",
      "output": "research/reproduction/a8/{a8_id}/vision_gates_{key}.json"
    },
    {
      "step": "input_delivery",
      "source": "PREREG_A2_DRAFT.md, Delivery checks",
      "output": "research/reproduction/a8/{a8_id}/input_delivery_{key}.json"
    },
    {
      "step": "readout_ladder_descriptive",
      "source": "PREREG_A2_DRAFT.md, Readouts; the readout is not re-chosen",
      "output": "research/reproduction/a8/{a8_id}/readout_ladder_{key}.json"
    },
    {
      "step": "a3",
      "source": "PREREG_A3_DRAFT.md",
      "output": "results/a8/{a8_id}/face_input_validity/cells_{key}.jsonl"
    },
    {
      "step": "a1",
      "source": "PREREGISTRATION.md, Amendment A1 and A1.1; DECISIONS 2026-11-08, A1 errored trials",
      "output": "results/a8/{a8_id}/rotation/eval_v2_eu_emotions_{a8_id}_video_only_seed42_rot{r}_{c}.json"
    },
    {
      "step": "natural_clip_rq1_descriptive",
      "source": "DECISIONS 2026-11-07, Natural-clip RQ1 under each RQ0 outcome",
      "output": "research/reproduction/a8/{a8_id}/"
    },
    {
      "step": "a2",
      "source": "PREREG_A2_DRAFT.md, with both dated deviations and the .01 sensitivity",
      "output": "results/a8/{a8_id}/stimulus_interventions/"
    },
    {
      "step": "a4",
      "source": "PREREG_A4_DRAFT.md",
      "output": "results/a8/a4/"
    }
  ],
  "a1": {
    "rotations": [
      0,
      1,
      2,
      3,
      4
    ],
    "conditions": [
      "real",
      "mismatch"
    ],
    "evaluate_args": {
      "dataset": "eu_emotions",
      "condition": "video_only",
      "stage": "stage2",
      "max_frames": 16,
      "fps": 1,
      "seed": 42,
      "n_options": 6,
      "data_root": "data/eu_emotions",
      "manifest": "data/eu_emotions_full_manifest.json",
      "human_options": "data/eu_emotions_human_entropy_rot{r}.json",
      "fc_samples": 20,
      "vision_mismatch_seed": 42
    },
    "sharding": "trials assigned by manifest position modulo n_shards; every shard loads the full manifest, so the frozen code computes mismatch partners over all items; only the scoring loop is restricted",
    "merge": "one file per rotation and condition, trials in manifest order, exact partition of the items required",
    "lookup_rule": "a missing or unpinned rotation lookup stops the run; there is no fallback to sampled foils"
  },
  "eligibility": {
    "rq0_a1": "PREREGISTRATION.md, Amendment A1, Decision, with the A8 family",
    "rq0_a3": "PREREG_A3_DRAFT.md, Estimand and decision; Preconditions; Missing data",
    "rq0_reading": "PREREG_A3_DRAFT.md, Reading against A1, as rq0_status in scripts/analyse_stimulus_interventions.py; a missing A1 or A3 result admits neither route",
    "gates": "G2, G3 and G4 must pass: PREREG_A2_DRAFT.md Depends on and Models; PREREG_A4_DRAFT.md Depends on; DECISIONS 2026-11-05 on dtype",
    "delivery": "PREREG_A2_DRAFT.md, Delivery checks",
    "a2": "PREREG_A2_DRAFT.md, Gate B, Decision rule per prediction, Null-result rule",
    "a4": "PREREG_A4_DRAFT.md, Depends on, Validity gate, Decision rule",
    "gemma4_31b_gates": "strict: if the frozen gemma4 adapter cannot run, no gate result and ineligible"
  },
  "stop_rules": {
    "skip_a1": "a model that fails G2 to G4, or whose A3 result is not Pass, does not run A1",
    "a2_a4_only_if_eligible": true,
    "storage": "no job starts above storage_stop"
  },
  "families": {
    "a3": "active models whose single_face delivery passed (input_delivery file, verdicts.single_face.pass); each must have a complete A3 cell file; a model whose delivery failed is not tested; a missing delivery file stops the analysis",
    "a1": "active models with G2 to G4 passed and A3 vision_dependent true, which are the models the stop rule lets run A1; each must have all 10 merged files; a model outside the family must have none",
    "a2": "Westfall-Young step-down maxT within one model and route, as frozen",
    "a4": "the frozen Holm over the models with A4 cells in results/a8/a4; only models eligible under A4's rules run A4",
    "pooling": "none, with frozen models, with each other, or with small siblings",
    "computed_by": "research/scripts/a8_family.py, from the A8 output files; no hand-picked model list in production"
  },
  "frozen_verdicts_unchanged": true,
  "natural_clip_rq1": "descriptive only",
  "queue": "after both Qwen A1 arrays are COMPLETED and all 30 rotation files exist, A8 is frozen, A8 code is pinned, and the frozen Phase 2 runs (Gate B and A2 on the eligible models, A4 runs) are COMPLETED",
  "equivalence_tests": {
    "a1_sharding_cpu": "the frozen run_evaluation with model, media and generation stubbed: all 243 items, mismatch, 4 shards, merged, identical to the unsharded run; plus 6 and 7 item runs in both conditions and sampling modes",
    "a1_sharding_gpu": "on Qwen3-VL-8B, rotation 0, 6 items, both conditions: the frozen command run twice, and the shard runner in 2 shards, merged",
    "a1_gpu_rule": "if the two frozen runs are identical, the merged shards must be identical to them, record for record; if the two frozen runs differ, A1 sharding is not shown and A1 on A8 models waits on a dated ruling made before any A8 A1 output exists",
    "a3_summary": "on synthetic cells equal to the frozen summarise output byte for byte; on the three frozen cell files (cluster) reproducing frozen_a3_summary_sha256",
    "a2_loader": "through a8_analyse_a2.main, a frozen A3 summary with one entry renamed to an A8 id gives output byte-identical to the frozen analysis"
  },
  "data_sha256": {
    "data/eu_emotions_full_manifest.json": "54e9976f1a05557b487a3965ba254656bc3b62ce34ba4449eb4e387518b5df7d",
    "data/eu_emotions_human_entropy.json": "f9d91b4e47816fd01b773f0a63f8fff39c7e0cfa2c56a2136b1f52ba5cc33744",
    "data/eu_emotions_human_entropy_rot0.json": "07ab51daf7630e67e5228233a6301ab2e9e0049df3da1ffc382fcd4a7620c996",
    "data/eu_emotions_human_entropy_rot1.json": "21b55baa90802567ab86fe65bc079109f36cbb94cd16ab7c738289e7ecef38b0",
    "data/eu_emotions_human_entropy_rot2.json": "ab6f67439e66faa7b2b5d9ab14acb081c0c4ec4c9427068c68b4ac1a7ee13e86",
    "data/eu_emotions_human_entropy_rot3.json": "84cad8f1bd8870605cf611c8cb438ac708394298d90b6f2a7458c20b74e33361",
    "data/eu_emotions_human_entropy_rot4.json": "c8c019b3f87a4ddc7779b2e05a0cb1e501b609b2da62190a33dbfc71df82e763",
    "research/reproduction/eu_au_measures.csv": "a148351851e4487ca83d0e5719dad25a99acd7a8d3ce8e00f8a77e4fa47db655",
    "research/reproduction/a4_v1_pilot.csv": "c735615a034c281909c96894c700d224638e281f55f995a0959eeef8a6fb3994",
    "results/a4/manifest.csv": "2ef1161e895224d499b360144690a07c5440bb018105e9c2ffc762f68badd2d1"
  },
  "data_sha256_note": "rotation lookups are the deterministic output of paper_v2.option_score write_rotated_human_lookup on the pinned human lookup, generated off-repo on 2026-11-09; the cluster copies are checked against these values by the launcher and are not yet hashed",
  "software_versions": {
    "pinned": {
      "torch": "2.10.0+cu128",
      "transformers": "5.10.1",
      "cv2": "4.11.0"
    },
    "recorded_only": [
      "python",
      "numpy"
    ],
    "source": "research/reproduction/vision_gates_qwen3vl.json and vision_gates_gemma4.json, field versions; numpy is UNVERIFIED for the sibling runs",
    "enforced_steps": [
      "gates",
      "a1_shard",
      "a3",
      "a4_run"
    ],
    "recorded_steps": [
      "delivery",
      "ladder",
      "a2_run"
    ],
    "rule": "every job records torch, transformers, cv2, numpy and python; an enforced step refuses to start on any difference from pinned; a recorded step runs in the frozen au_tools overlay, whose sibling versions were not recorded (UNVERIFIED), and its versions are reported"
  },
  "frozen_outputs_guard": [
    "results/baseline/eu_emotions/{key}/*",
    "research/reproduction/*_{key}.json",
    "research/reproduction/rotation_summary.json",
    "research/reproduction/face_input_validity_summary.json",
    "research/reproduction/same_clip_family_model_{key}.json",
    "results/stimulus_interventions/*{key}*",
    "results/face_input_validity/*",
    "results/a4/cells_{key}_*.jsonl",
    "results/a4/summary.json"
  ],
  "commands": {
    "gates": {
      "module": "scripts.vision_gates",
      "argv": [
        "--model",
        "{key}",
        "--gates",
        "g2,g3,g4",
        "--dtypes",
        "bfloat16,float32",
        "--n_g3",
        "12",
        "--output",
        "research/reproduction/a8/{a8_id}/vision_gates_{key}.json"
      ],
      "outputs": [
        "research/reproduction/a8/{a8_id}/vision_gates_{key}.json"
      ],
      "resumable": false,
      "env": "conda_cv411",
      "decoder_preflight": true,
      "uses_model": true,
      "argv_extra_by_key": {
        "qwen3vl": [
          "--float32_max_frames",
          "6"
        ],
        "gemma4": []
      },
      "argv_note": "as slurm_jobs/study3_vision_gates.sh: --float32_max_frames 6 for qwen3vl only, after --output"
    },
    "delivery": {
      "module": "scripts.run_stimulus_interventions",
      "argv": [
        "--step",
        "delivery",
        "--model",
        "{key}",
        "--geometry",
        "results/stimulus_interventions/geometry.json",
        "--max_items",
        "3",
        "--reproduction",
        "research/reproduction/a8/{a8_id}",
        "--output_dir",
        "results/a8/{a8_id}/stimulus_interventions"
      ],
      "outputs": [
        "research/reproduction/a8/{a8_id}/input_delivery_{key}.json"
      ],
      "resumable": false,
      "env": "au_tools",
      "decoder_preflight": true,
      "uses_model": true
    },
    "ladder": {
      "module": "scripts.readout_ladder",
      "argv": [
        "--step",
        "ladder",
        "--model",
        "{key}",
        "--rungs",
        "1,2,3",
        "--output",
        "research/reproduction/a8/{a8_id}"
      ],
      "outputs": [
        "research/reproduction/a8/{a8_id}/readout_ladder_{key}.json"
      ],
      "resumable": false,
      "env": "au_tools",
      "decoder_preflight": true,
      "uses_model": true
    },
    "a3": {
      "module": "scripts.face_input_validity",
      "argv": [
        "--model",
        "{key}",
        "--a3_frozen",
        "--output_dir",
        "results/a8/{a8_id}/face_input_validity"
      ],
      "outputs": [
        "results/a8/{a8_id}/face_input_validity/cells_{key}.jsonl"
      ],
      "resumable": true,
      "env": "conda_cv411",
      "decoder_preflight": true,
      "uses_model": true
    },
    "a1_shard": {
      "runner": "research/scripts/a8_rotation_shard.py run",
      "outputs": [
        "results/a8/{a8_id}/rotation/shards/rot{rotation}_{condition}/shard{shard}of{n_shards}/eval.json"
      ],
      "resumable": false,
      "env": "conda_cv411",
      "decoder_preflight": true,
      "uses_model": true
    },
    "a1_merge": {
      "runner": "research/scripts/a8_rotation_shard.py merge",
      "outputs": [
        "results/a8/{a8_id}/rotation/eval_v2_eu_emotions_{a8_id}_video_only_seed42_rot{rotation}_{condition}.json"
      ],
      "resumable": false,
      "env": "any",
      "decoder_preflight": false,
      "uses_model": false
    },
    "natural_rq1": {
      "path": "research/scripts/same_clip_family_analysis.py",
      "argv": [
        "--model-route",
        "--au-csv",
        "research/reproduction/eu_au_measures.csv",
        "--model",
        "{a8_id}",
        "--model-evals",
        "results/a8/{a8_id}/rotation/eval_v2_eu_emotions_{a8_id}_video_only_seed42_rot*_real.json"
      ],
      "declared_output_outside_roots": "research/reproduction/same_clip_family_model_{a8_id}.json",
      "outputs": [
        "research/reproduction/a8/{a8_id}/same_clip_family_model_{a8_id}.json"
      ],
      "resumable": false,
      "env": "any",
      "decoder_preflight": false,
      "uses_model": false
    },
    "a2_run": {
      "module": "scripts.run_stimulus_interventions",
      "argv": [
        "--step",
        "run",
        "--model",
        "{key}",
        "--geometry",
        "results/stimulus_interventions/geometry.json",
        "--reproduction",
        "research/reproduction/a8/{a8_id}",
        "--output_dir",
        "results/a8/{a8_id}/stimulus_interventions",
        "--inputs",
        "single_face,single_full,video_face,legacy_full",
        "--shard",
        "{shard}",
        "--n_shards",
        "{n_shards}",
        "--confirm_gates_passed"
      ],
      "outputs": [
        "results/a8/{a8_id}/stimulus_interventions"
      ],
      "resumable": true,
      "env": "au_tools",
      "decoder_preflight": true,
      "uses_model": true
    },
    "a4_run": {
      "module": "scripts.a4_run",
      "argv": [
        "--model",
        "{key}",
        "--shard",
        "{shard}",
        "--n_shards",
        "{n_shards}",
        "--a4_frozen",
        "--output_dir",
        "results/a8/a4"
      ],
      "outputs": [
        "results/a8/a4"
      ],
      "resumable": true,
      "env": "conda",
      "decoder_preflight": false,
      "uses_model": true
    },
    "a4_analyse": {
      "module": "scripts.a4_analyse",
      "argv": [
        "--cells_dir",
        "results/a8/a4",
        "--gates",
        "research/reproduction/a8/a4_gates.json",
        "--detector",
        "research/reproduction/a4_v1_pilot.csv",
        "--output",
        "results/a8/a4/summary.json"
      ],
      "outputs": [
        "results/a8/a4/summary.json"
      ],
      "resumable": false,
      "env": "any",
      "decoder_preflight": false,
      "uses_model": false
    }
  },
  "rules": {
    "oom": "an out-of-memory error recorded in a trial (A1 trials[].error), a cell (A3, A2, A4 cells: reason, error or detail) or a delivery row (failed) fails its job: the output is not merged or analysed, and the job is rerun with more GPUs before any analysis. Read from those structured fields only. The gate file is not scanned: the frozen gate script records the expected float32 out-of-memory error in float32_failed and keeps the bfloat16 verdict (DECISIONS 2026-11-05); a gate job that ends without its file is a failed job",
    "errored_not_oom": "scored as the frozen pipeline scores it (wrong, no vote), kept in the merge, and counted per model, condition and rotation",
    "gpu_settings": "all shards of a rotation and condition, and the real and mismatch tasks of a model, run with the same visible GPU count; merge and the A1 family check refuse otherwise",
    "scored_runs_need_code_pins": true,
    "test_mode_outputs": "a shard or job run in test mode is never merged into, or analysed from, results/a8/{a8_id}",
    "not_tested_after_retry": "a per-model step (gates, delivery, ladder, a3) whose launcher record shows the job started and did not end ok, twice (the run and one retry, at the same run settings or more GPUs), with no output file, counts as not tested for that model: no gate result (not passed), single_face delivery failed (A3 not tested), or no A3 result; the family rules then exclude the model and the other models' analyses go on",
    "eligibility_checked_by_launcher": "before a1_shard: G2 to G4 passed and A8 A3 vision_dependent true; before a2_run: G2 to G4 passed, single_face delivery passed and the frozen rq0_status(a1, a3, single_face) true from the A8 A1 and A3 summaries; before a4_run: G2 to G4 passed and rq0_status(a1, a3, single_face) true; summaries accepted only with their provenance; the decision is recorded; an ineligible model is refused",
    "inputs_from_launched_jobs": "every analysis reads model outputs only from the paths the commands fix for that a8_id, and only when a launcher record with status ok and the design's weights path exists for the job that wrote them (A1: the shard sidecars' resolved_model_path)"
  },
  "data_by_step": {
    "gates": [
      "data/eu_emotions_full_manifest.json",
      "data/eu_emotions_human_entropy.json"
    ],
    "delivery": [
      "data/eu_emotions_full_manifest.json",
      "data/eu_emotions_human_entropy.json"
    ],
    "ladder": [
      "data/eu_emotions_full_manifest.json",
      "data/eu_emotions_human_entropy.json"
    ],
    "a3": [
      "data/eu_emotions_full_manifest.json",
      "data/eu_emotions_human_entropy.json"
    ],
    "a1_shard": [
      "data/eu_emotions_full_manifest.json",
      "data/eu_emotions_human_entropy.json",
      "data/eu_emotions_human_entropy_rot{rotation}.json"
    ],
    "a2_run": [
      "data/eu_emotions_full_manifest.json",
      "data/eu_emotions_human_entropy.json"
    ],
    "a4_run": [
      "results/a4/manifest.csv"
    ],
    "natural_rq1": [
      "research/reproduction/eu_au_measures.csv",
      "data/eu_emotions_human_entropy.json"
    ],
    "a4_analyse": [
      "research/reproduction/a4_v1_pilot.csv"
    ]
  }
}
```

## How every A8 job runs

One launcher, `research/scripts/a8_launch.py`, is the only entry point. `slurm_jobs/a8_job.sh` sets up, line for line, the environment the frozen job of the same step sets up for a non-Molmo model (conda with the cv2 4.11 overlay for A1, gates and A3; the au_tools venv alone for delivery, ladder and A2; conda for A4), and never sets `MR_EU_HPC_MODELS_DIR`; the remaining differences are in Limits. The launcher, in one process:

1. loads the block (hash-checked), checks every frozen file against `frozen_code_sha256` and A8's code against "Code hashes" (required);
2. builds the frozen command line from the block's `commands` entry, so the arguments the frozen script receives are the ones checked;
3. for a model step, sets `MR_EU_HPC_MODELS_DIR` from the block (refusing a different preset value), runs the weights preflight, imports `config` and `scripts.evaluate`, and asserts that both `config.MODELS[key]["hpc_path"]` and `evaluate.resolve_model_path(key)` resolve to the block's weights directory, recording both;
4. checks the step's pinned data files (block `data_by_step`, which covers every model step), the storage stop rule and the software versions (enforced where the block says), and runs the frozen `decoder_preflight.py` where the frozen job runs it; before `a1_shard`, `a2_run` and `a4_run` it checks eligibility by the frozen rules from the A8 files (block rule `eligibility_checked_by_launcher`), records the decision, and refuses an ineligible model;
5. requires every output under the A8 roots, and a non-resumable output not to exist yet;
6. hashes every frozen output of the sibling key (`frozen_outputs_guard`) before and after the job, and fails the job if any changed;
7. writes a "started" record, runs the frozen module with `runpy` (or the A1 shard runner, or the merge), then fails the job if a trial, cell or delivery-row error field records an out-of-memory error (structured fields only; the gate file is not scanned, since the frozen gate script records its expected float32 out-of-memory error in `float32_failed`);
8. writes a provenance JSON under `research/reproduction/a8/<a8_id>/launch/`.

The commands, from the block, with `<id>` the A8 id and `<key>` its code key:

| Step | Frozen source | Frozen code and arguments | Output |
|---|---|---|---|
| gates | DECISIONS 2026-11-04, 2026-11-05; `GATE_RULES` | `scripts.vision_gates --model <key> --gates g2,g3,g4 --dtypes bfloat16,float32 --n_g3 12 --output research/reproduction/a8/<id>/vision_gates_<key>.json`, plus `--float32_max_frames 6` for `qwen3vl` only, as the frozen job passes it | that file |
| delivery | `PREREG_A2_DRAFT.md`, Delivery checks | `scripts.run_stimulus_interventions --step delivery --model <key> --geometry results/stimulus_interventions/geometry.json --max_items 3 --reproduction research/reproduction/a8/<id> --output_dir results/a8/<id>/stimulus_interventions` | `research/reproduction/a8/<id>/input_delivery_<key>.json` |
| ladder (descriptive) | `PREREG_A2_DRAFT.md`, Readouts; the readout is not re-chosen | `scripts.readout_ladder --step ladder --model <key> --rungs 1,2,3 --output research/reproduction/a8/<id>` | `research/reproduction/a8/<id>/readout_ladder_<key>.json` |
| a3 | `PREREG_A3_DRAFT.md` | `scripts.face_input_validity --model <key> --a3_frozen --output_dir results/a8/<id>/face_input_validity` (resumes) | `results/a8/<id>/face_input_validity/cells_<key>.jsonl` |
| a1_shard | A1, A1.1; DECISIONS 2026-11-08 | `a8_rotation_shard.py run`, which calls the frozen `evaluate.main()` with the frozen rotation arguments | `results/a8/<id>/rotation/shards/rot<r>_<c>/shard<k>of<n>/eval.json` and its sidecar |
| a1_merge | | `a8_rotation_shard.py merge` | `results/a8/<id>/rotation/eval_v2_eu_emotions_<id>_video_only_seed42_rot<r>_<c>.json` |
| natural_rq1 (descriptive) | DECISIONS 2026-11-07 | `same_clip_family_analysis.py --model-route --au-csv research/reproduction/eu_au_measures.csv --model <id> --model-evals '<merged real files>'` | moved to `research/reproduction/a8/<id>/same_clip_family_model_<id>.json` |
| a2_run | `PREREG_A2_DRAFT.md` | `scripts.run_stimulus_interventions --step run --model <key> --geometry results/stimulus_interventions/geometry.json --reproduction research/reproduction/a8/<id> --output_dir results/a8/<id>/stimulus_interventions --inputs single_face,single_full,video_face,legacy_full --shard <k> --n_shards <n> --confirm_gates_passed` (resumes) | `results/a8/<id>/stimulus_interventions/` |
| a4_run | `PREREG_A4_DRAFT.md` | `scripts.a4_run --model <key> --shard <k> --n_shards <n> --a4_frozen --output_dir results/a8/a4` (resumes) | `results/a8/a4/` |
| a4_analyse | `PREREG_A4_DRAFT.md` | `scripts.a4_analyse --cells_dir results/a8/a4 --gates research/reproduction/a8/a4_gates.json --detector research/reproduction/a4_v1_pilot.csv --output results/a8/a4/summary.json` | that file |

Analyses outside the launcher, each run once, each refusing to overwrite, each reading model outputs only from the paths the commands fix and only when a launcher record with status ok and the design's weights path exists for the job that wrote them (block rule `inputs_from_launched_jobs`; for A1, the shard sidecars' resolved path), and each writing a provenance file (output and input hashes) that the next consumer checks: `a8_face_input_validity_summary.py` (A3), `a8_analyse_rotation.py` (A1), `a8_a4_gates.py` (the gates file A4 reads), `a8_analyse_a2.py` (A2). With one A8 model per key in `results/a8/a4`, the frozen `a4_analyse` Holm over `d.MODELS` is the A8 A4 family.

A5, A6 and A7 do not run on A8 models.

## Eligibility: the frozen rules only

No new threshold is set.

1. **RQ0, A1.** `PREREGISTRATION.md`, "Amendment A1", Decision: vision-dependent if the one-sided actor-level sign-flip p (100,000 draws) has Holm p < .05 across the family and the actor-bootstrap 95% CI (10,000 draws) of delta excludes 0 from above. The family is A8's.
2. **RQ0, A3.** `PREREG_A3_DRAFT.md`, "Estimand and decision", "Preconditions" (the `single_face` delivery gate must pass, or the model is not tested), "Missing data".
3. **Reading RQ0.** `PREREG_A3_DRAFT.md`, "Reading against A1", as applied by `rq0_status`: A3 Pass with A1 Pass admits both routes; A3 Pass with A1 Fail admits `single_face` only; anything else, including a missing result, admits neither.
4. **Vision gates.** G2, G3 and G4 pass, as required by `PREREG_A2_DRAFT.md` ("Depends on", "Models") and `PREREG_A4_DRAFT.md` ("Depends on"), with `GATE_RULES` and the DECISIONS 2026-11-05 dtype ruling. A gate that does not run has not passed.
5. **Delivery.** `PREREG_A2_DRAFT.md`, "Delivery checks".
6. **A2.** Gate B, "Decision rule per prediction", "Null-result rule", both dated deviations and the step-down p < .01 sensitivity.
7. **A4.** "Depends on", "Validity gate, before any human comparison", "Decision rule, per model and dataset".

**Stop rule.** A model that fails G2 to G4, or whose A3 result is not Pass, cannot become eligible for A2 or A4 whatever A1 shows. It does not run A1, and A8 reports "A1 not run, by the A8 stop rule". This does not depend on any A1 result.

## Statistics

A8 forms its own families, of A8 models only, computed by `research/scripts/a8_family.py` from the A8 output files, with no hand-picked list (block `families`):

- **A3:** active models whose `single_face` delivery passed; each needs a complete cell file (243 items by 10 cells, unique keys, no partial line). A failed delivery is "not tested"; a missing delivery file stops the analysis.
- **A1:** active models with G2 to G4 passed and A3 Pass, which are the models the stop rule lets run A1; each needs all 10 merged files with one GPU count and no test-mode shard; a model outside the family must have none.
- **A2:** Westfall-Young within model and route, as frozen.
- **A4:** the frozen Holm over the models with A4 cells in `results/a8/a4`; only models eligible under A4's rules run A4.

The frozen families are untouched. With one model in an A8 family, Holm is the identity, as in the frozen code with one model; with two, Holm over two. Holm controls familywise error at .05 at any size; a smaller family is less strict per model than the frozen three-model family, and the paper says so.

**No pooling** with frozen models, with each other, or with the small siblings.

**Reporting.** A8 rows are labelled A8 and placed under the frozen rows, whatever they show. Where an A8 model and its sibling disagree, both verdicts are reported and the difference is described, not tested. A8 may support a scope statement, never a change to a frozen verdict.

**Cannot rescue.** No frozen verdict is changed by A8. A frozen Inconclusive stays Inconclusive, a frozen null stays a null, and Molmo2 stays ineligible.

**What is read.** Each A8 analysis runs once, after all its outputs exist. Mechanics and timing smokes are read for status, seconds, memory and forwards only.

## Errors, memory and run settings

- **Errored trials and cells** (any error except out of memory) are scored as the frozen pipeline scores them (an errored A1 trial is wrong and casts no vote, DECISIONS 2026-11-08), kept in the merge, and counted per model, condition and rotation.
- **Out of memory.** Any trial, cell or delivery row whose error field records an out-of-memory error fails its job: the shard sidecar is written, the job exits non-zero, the merge refuses the shard, and the job is rerun with more GPUs before any merge or analysis. This keeps memory headroom, which depends on GPU count, from deciding which trials score.
- **Gates.** The frozen gate script catches the float32 stage's out-of-memory error, records it in `float32_failed`, and keeps the bfloat16 verdict; that is not a job failure. A gate job that ends without its file is a failed job.
- **Not tested after one retry.** A per-model step (gates, delivery, ladder, A3) whose launcher records show two attempts, none ending ok, and no output file, counts as not tested for that model (block rule `not_tested_after_retry`): no gate result, delivery failed (A3 not tested), or no A3 result. The family rules then exclude that model, and the other model's analyses go on.
- **GPU settings.** All shards of a rotation and condition, and the real and mismatch tasks of a model, run with the same visible GPU count; the merge and the A1 family check refuse otherwise.
- **Run settings** (shard counts, GPU counts) cannot change any successfully scored cell: A1 by the equivalence tests, A2 and A4 because their frozen analyses accept any complete set of shards. They are set by dated notes after the timing smokes.

## Compute

GPU: NVIDIA A100-SXM4-80GB on `ampere` (`gpu:nvidia_a100:4` per node; GPU name stored in `vision_gates_*.json`). Account `baron-cohen-sl3-gpu` is associated with partition `ampere` and QOS `gpu2,intr` only (`sacctmgr show assoc`, 2026-11-09); `gpu2` has a 12 h wall limit. `ampere-long` allows only QOS `gpul` (`scontrol show partition ampere-long`), which this account does not have.

**Assumptions.** Time scales with language-model parameters: 33.36 / 8.77 = 3.80 for Qwen3-VL-32B over 8B (Gemma 4 31B over E4B: 3.9 by stored parameters, up to about 7.8 if E4B computes as about 4 B, UNVERIFIED). Upper-end, since the vision tower is the same size on both Qwen models and decoding and data loading do not scale. One GPU per job; a step that needs two doubles its GPU-hours.

| Step | Base, measured | Qwen3-VL-32B, GPU-hours | Gemma 4 31B |
|---|---|---|---|
| Mechanics and timing smokes, plus the A1 equivalence test | none | about 4 | about 3 |
| Vision gates | Qwen 1.74 h; Gemma 0.09 h (bfloat16 stage) | 6.6, plus about 0.5 | 0.4 to 0.7, plus about 0.5 |
| Input delivery | not recorded | about 1 (UNVERIFIED) | about 1 (UNVERIFIED) |
| Readout ladder | Qwen 0.30 h; Gemma 0.18 h | 1.1 | 0.7 to 1.4 |
| A3 | Qwen 1.12 h; Gemma 0.82 h | 4.3 | 3.2 to 6.4 |
| **Phase A** | | **about 17** | **about 9 to 13** |
| A1, 10 tasks | Qwen 7.73 h per task; Gemma E4B not in the repo | about 29.4 per task, 294 in all | 3.9 T to 7.8 T per task (T = E4B's task time, from Slurm accounting) |
| Natural-clip RQ1 | CPU only | 0 | 0 |
| A2, with 10% allowance | Qwen 16.5 h; Gemma 11.6 h | 69 | 50 to 100 |
| A4 | A4's upper bound about 61 and 60 h; from A3's per-cell time about 36 and 26 h | 141 to 232 | 103 to 468 |
| **Total if eligible at every step** | | **about 520 to 610** | **about 160 to 580, plus A1** |

A8 runs in phases: Phase A, then A1 where the stop rule allows, then A2 and A4 where eligible. If a measured phase projects Qwen3-VL-32B over 600 GPU-hours in total, A4 is the step deferred, by a dated note before it runs. GPU-hours available: 2,859 (STATE.md item 1, 2026-11-09).

**A1 sharding.** `scripts/evaluate.py` cannot resume, and a 32B task projects to about 29 h against the 12 h limit. Splitting the manifest is not allowed: `evaluate.py` computes mismatch partners over the trials it loads, and reads a partner's video from that set. `research/scripts/a8_rotation_shard.py` runs the frozen `evaluate.main()` on the full manifest, so the frozen `vision_mismatch_partners` builds the map over all 243 items as in the unsharded run, and restricts only the scoring loop, by giving `scripts.evaluate` a module-level `enumerate` that yields the shard's `(index, trial)` pairs, with their original indices, when applied to the loaded trial list. Draws are seeded per trial and the option set uses the original index, so a record does not depend on its shard. The runner also checks every scored mismatch record against the frozen full-set map, keeps errored records (which carry no partner field), refuses a missing or unpinned lookup, and records the resolved weights path, GPU count, versions and arguments in a sidecar that the merge compares across shards. Four shards per task put a 32B shard at about 7.4 h.

**Timing.** No A8 job is submitted until both Qwen A1 arrays show COMPLETED for every task, all 30 rotation files exist, A8 is frozen and its code pinned, and the frozen Phase 2 runs (Gate B and A2 on the eligible models, A4 runs) are COMPLETED. The launcher requires `--confirm_phase2_completed` and records it; the frozen-output guard makes an overlap detectable even if the confirmation were wrong.

## Code

No existing file is edited. New files, written 2026-11-09, to be pinned in "Code hashes":

| File | Purpose |
|---|---|
| `research/scripts/a8_design.py` | Block hash, frozen-code pins, data pins, A8 code pins, active models, import closure |
| `research/scripts/a8_family.py` | A3 and A1 families by rule, A3 completeness, A1 run settings, provenance |
| `research/scripts/a8_preflight.py` | Weights, environment variable, outputs, storage; `--full_hash` |
| `research/scripts/a8_launch.py` | The one entry point for every A8 job |
| `research/scripts/a8_rotation_shard.py` | A1 shard runner, merge, compare, equivalence runs |
| `research/scripts/a8_face_input_validity_summary.py` | A3 summary over the A8 A3 family |
| `research/scripts/a8_analyse_rotation.py` | A1 analysis over the A8 A1 family |
| `research/scripts/a8_analyse_a2.py` | A2 analysis with the one loader swap |
| `research/scripts/a8_a4_gates.py` | The `--gates` file for the frozen A4 analysis |
| `slurm_jobs/a8_job.sh` | Environment per step, then the launcher |
| `tests/test_a8.py` | Unit tests and the three equivalence tests; GPU and frozen-file tests skip without them |
| `research/scripts/a8_prerun_check.py` | Added 2026-11-13, before any A8 output (dated note of that day): the A2 and A4 pre-run checks on A8's own inputs. It loads `research/scripts/a2_prerun_check.py` and `research/scripts/a4_prerun_check.py`, and the A2 step runs `research/scripts/a2_declared_extras.py`; those three are pinned with A8's code |

## Code hashes

```text
pending: pinned by a dated, pre-data step before any A8 output exists
```

## Recovery

Not needed. Recovery in A2 and A4 checks that the analysis recovers known observers on the real items, actors, contrasts and constants; it does not depend on which model produced the cells. A8 uses the same analysis code, items, actors, contrasts, constants and design hashes. A1 and A3 had no recovery step. What is new is checked by equivalence tests: sharding (identical records, including all 243 items with the full partner map, on the frozen code path with stubs, and on Qwen3-VL-8B on the cluster), the A3 model-set wrapper (identical bytes), and the A2 loader swap (identical bytes through the production path). Holm on a smaller family needs no recovery.

## Audit rulings, 2026-11-09

Audit: `research/reviews/a8_audit.md`. "Fixed" means fixed in the block, code or tests, with the test named where one was added (all in `tests/test_a8.py`).

| # | Severity | Finding | Ruling |
|---|---|---|---|
| 1 | High | An errored mismatch trial made a shard unmergeable | Fixed. Errored records are skipped in the partner check, must carry no partner field, are kept and merged, and score as frozen. `test_an_errored_trial_is_kept_scored_wrong_and_merged` (both conditions) |
| 2 | High | Nothing tied checked weights to loaded weights | Fixed. One launcher sets `MR_EU_HPC_MODELS_DIR` from the block, refuses a different preset value, asserts `config` and `resolve_model_path` both resolve to the block's weights directory, and records them; the shard runner does the same and the merge compares the resolved path; the slurm file unsets the variable. `test_preflight_requires_the_environment_variable_to_match`, `test_models_dir_variable_is_set_from_the_design_and_a_different_value_refused` |
| 3 | High | Step commands could overwrite frozen outputs | Fixed. Every command, with explicit output arguments, is in the block and is what the launcher runs; natural-clip RQ1's fixed output is a declared exception, refused if present and moved into the A8 folder; frozen outputs of the key are hashed before and after each job; queue on Phase 2 COMPLETED. `test_every_step_writes_only_under_the_a8_roots_with_explicit_paths`, `test_frozen_output_guard_detects_any_change` |
| 4 | High | A1 environment not reproduced; missing lookup fell back silently | Fixed. Decoder preflight and the cv2 4.11 pin (pinned versions) run in production; a missing lookup is refused; manifest, human lookup, the five rotation lookups and two other inputs are pinned in the block. `test_a_missing_lookup_is_refused_not_replaced`, `test_data_inputs_are_pinned_and_the_lookups_are_reproducible` |
| 5 | Medium | Family code did not implement the text | Fixed. Families are computed from the A8 files by `a8_family.py`; no hand-picked list in production. `test_a3_family_is_computed_from_delivery_files`, `test_a1_family_follows_the_stop_rule_and_requires_every_file` |
| 6 | Medium | Admission lived outside the hash | Fixed. Gemma 4 31B is `admitted_2026-11-09_user_approved_storage` inside the block; the note-reading path is removed. `test_both_models_are_active_by_the_block_alone` |
| 7 | Medium | Loaded modules and software versions unpinned | Fixed. All 42 repo modules in the import closure of the entry points are pinned, with a completeness test; torch 2.10.0+cu128, transformers 5.10.1 and cv2 4.11.0, as recorded by the siblings' gate runs, are pinned and enforced for gates, A1, A3 and A4 runs; numpy and the au_tools steps' versions are recorded only (UNVERIFIED for the siblings). `test_every_module_reached_by_import_is_pinned` |
| 8 | Medium | Equivalence test missed the full-243 partner map | Fixed locally: `test_a1_equivalence_on_all_243_items_with_the_full_partner_map`. Cluster test runs the frozen command twice; the block fixes the rule if the two differ (stop, dated ruling). The optional comparison against the 8B's frozen rotation file is not done, because it would read an A1 result |
| 9 | Medium | Runner not bound to the design | Fixed. `--a8_id` fixes key, sampling, directories and output path; `--equivalence_label` is separate; code pins required in every scored entry point; `--test_mode` recorded and never merged into `results/a8`. `test_the_runner_is_bound_to_the_design_per_model` |
| 10 | Medium | A2 test near tautology; A2 inputs unpinned | Fixed. The test runs the production path (`run`) with a renamed summary and requires byte equality; A3 and A1 summaries are accepted only with matching provenance; `--model` and `--reproduction` are set by the wrapper. `test_a2_production_path_gives_the_frozen_output_byte_for_byte`, `test_a2_wrapper_refuses_hand_set_paths_and_unrecorded_inputs` |
| 11 | Medium | Incomplete A3 cell file accepted | Fixed. `test_an_incomplete_a3_cell_file_is_refused` |
| 12 | Medium | Run settings could change errored trials | Fixed by rule: out-of-memory fails the job and is rerun; identical GPU settings across shards and conditions; wording now "successfully scored cell". `test_an_out_of_memory_trial_fails_the_shard_and_blocks_the_merge`, `test_out_of_memory_in_any_output_fails_the_job` |
| 13 | Low | Fixed sidecar name | Fixed: `<stem>.a8_shard.json` and a fresh directory per shard. `test_each_shard_needs_a_fresh_directory` |
| 14 | Low | Merge keys incomplete | Fixed: arguments without output, resolved path, GPU count, versions and test mode compared |
| 15 | Low | Relative paths against ROOT vs cwd | Fixed: every A8 entry point changes to the project root first |
| 16 | Low | `--skip_storage` unrecorded | Fixed: the option is removed; the launcher always checks storage |
| 17 | Low | A1 wrapper gaps | Fixed: refuses an existing sidecar; always loads the design and checks frozen code; no `--models` |
| 18 | Low | A3 frozen-file equivalence not run | Ruled: a cluster item, added to the freeze checklist (item 3); the frozen cell files are not in the local checkout and the cluster is not touched today |
| 19 | Low | float32 ruling generalised | Ruled: stated in Limits |
| 20 | Low | Weak A8 path check | Fixed: A3 paths are built by `a8_family`, not given |
| 21 | Low | Weak tests | Fixed: replaced as listed above, plus `test_a1_analysis_runs_the_frozen_main_over_the_a8_family` |

**Re-check, 2026-11-09** (appended to the audit file):

| # | Severity | Finding | Ruling |
|---|---|---|---|
| R1 | High, blocker | Gates would always end "out_of_memory"; `--float32_max_frames 6` missing for qwen3vl | Fixed in the block and code: `commands.gates.argv_extra_by_key` adds `--float32_max_frames 6` for `qwen3vl` after `--output`, as the frozen job; `rules.oom` is narrowed to trial, cell and delivery-row error fields; the scan reads structured fields only, skips `float32_failed` and raw model text, and does not scan the gate file. `test_the_frozen_8b_gate_file_is_not_flagged`, `test_gates_command_mirrors_the_frozen_job_for_each_key`, `test_out_of_memory_in_a_cell_error_field_fails_the_job` |
| R2 | Medium | No eligibility check before A1, A2, A4 runs | Fixed: block rule `eligibility_checked_by_launcher`; `a8_family.eligibility` (frozen `rq0_status`, gates, delivery, provenance-checked summaries) runs in the launcher and the decision is recorded. `test_the_launcher_eligibility_follows_the_frozen_rules` |
| R3 | Medium | A2 wrapper accepted any cells and reports; abbreviations could override | Fixed: cells and reports are built only from `results/a8/<id>/stimulus_interventions/`, each shard backed by an ok launcher record with the design's weights path, as are the gate and delivery files; the wrapper's own options, and any abbreviation of them or of `--output`, are refused; its parser uses `allow_abbrev=False`. The A3 producer requires an ok `a3` launcher record with the design's weights; the A1 producer requires every shard sidecar to be a production run on the design's weights. Block rule `inputs_from_launched_jobs`. `test_a2_wrapper_refuses_hand_set_paths_abbreviations_and_unrecorded_inputs`, `test_a3_inputs_need_a_launcher_record_with_the_design_weights`, `test_a2_production_path_gives_the_frozen_output_byte_for_byte` |
| R4 | Medium | A missing delivery file blocked every model | Fixed: block rule `not_tested_after_retry`; the launcher writes a "started" record so a killed job counts as an attempt; `a3_family` applies the rule. `test_a_step_that_fails_twice_counts_as_not_tested_and_others_go_on` |
| R5 | Low | Environment differed from the frozen jobs | Fixed: `a8_job.sh` mirrors each frozen job's activation and variables (au_tools venv alone for delivery, ladder and A2; frozen `HF_HOME`; no added variables). Remaining differences in Limits |
| R6 | Low | Data check missed gates and a4_run; plain-text OOM scan | Fixed: block `data_by_step` covers every model step (A4's pinned manifest added); structured scan as in R1. `test_every_model_step_checks_its_data_inputs` |
| 18 (re-check) | | A3 frozen-file equivalence must precede the freeze tag | Fixed: freeze checklist item 1 requires it. The frozen A3 cell files were fetched locally (sha256 `a3f302b3...`, `05eec2d4...`, `cd80b5c5...`, as RESULTS_REGISTRY records) and `test_a3_wrapper_on_the_frozen_cell_files` passes: the wrapper equals the frozen `--summarise` run locally, byte for byte. Against the pinned summary (made on the cluster) every field is equal except the last digits of one descriptive p, `secondary.accuracy_real_vs_chance_pooled_p` (Gemma 1.431505210601853e-65 against 1.4315052106018534e-65; Qwen 6.0602500041013055e-24 against 6.0602500041016956e-24), which A3 itself calls descriptive. That comes from the library build (local scipy 1.15.3, numpy 2.1.3; the cluster versions were not recorded), not from the wrapper. The exact hash is checked in the frozen environment (`A8_FROZEN_ENV=1` on the cluster) |

## Known limits before code pins

- The rotation lookups are pinned from an off-repo regeneration with the frozen writer; the cluster copies were not hashed today (no cluster access by rule). If they differ, the launcher refuses and the difference is resolved by a dated note before any A1 job.
- The au_tools steps (delivery, ladder, A2) run in the overlay the frozen jobs used, whose library versions were not recorded for the siblings; their versions are recorded, not enforced.
- The launcher has not run end to end (it needs the cluster, the weights and the code pins); its parts are unit-tested. Its first use is the dated pin step's mechanics run.
- The A1 equivalence test on the real 8B is a cluster item (part of the code-pin step).
- Timing smokes are not yet run, so run settings and the cost table are planning values.

## Freeze checklist

1. Design block audited and re-checked (done 2026-11-09), hash stated above and pinned in `a8_design.py`. **Precondition of the tag:** `test_a3_wrapper_on_the_frozen_cell_files` passes on the three frozen A3 cell files (done locally 2026-11-09, with the last-digit difference recorded in "Audit rulings"), and the exact-hash form (`A8_FROZEN_ENV=1`) passes in the frozen environment on the cluster, as the block claims; both results are recorded. Then tag `a8-freeze` on the design text before the Qwen3-VL-8B A1 result is read; DECISIONS entry.
2. Dated pre-data step, before any A8 output: code pins filled; the A1 GPU equivalence test (frozen twice, shards, merge) run and recorded; timing smokes read for mechanics only; run settings set.
3. On the cluster: rotation lookups compared with the pins (the A3 frozen-file test moved to item 1).
4. `a8_preflight.py --full_hash` on both weight copies; every sha256 equal to the block.
5. Gemma 4 E4B A1 task time read from Slurm accounting.
6. Frozen code on the cluster equal to `frozen_code_sha256` after the Phase 2 sync. On 2026-11-09 every A1 file matched; `scripts/run_stimulus_interventions.py` differed (cluster `19b1fdae...`, pre-sync, expected per ROADMAP 2.1).
7. Storage check at submission (the launcher also checks it).

## Freeze (2026-11-09, tag `a8-freeze`)

- Design hash `95dd41b5d3c33b79ee9e54a20d34ea1e18b6c9156fcabd719ebfb084709b7b50`, recomputed equal from the block and pinned in `research/scripts/a8_design.py`.
- Audit: first pass 21 findings, re-check R1 to R6, final check freeze-ready (`research/reviews/a8_audit.md`).
- Item 1 done. Locally, `tests/test_a8.py` 54 passed, 1 skipped (the GPU A1 equivalence test, item 2). On the cluster, in the frozen conda env `mr_eu_open_llm` (numpy 2.2.6, scipy 1.15.3), the A3 wrapper on the frozen cell files gave sha256 `c1c64e6975b32356a3314268c054f0dee483ddf74b11cd66d03b2faf972e64a4`, equal to the pinned frozen summary, 2026-11-09 about 10:45, run from a scratch copy (`~/rds/hpc-work/a8_check`) so no file in the project tree changed.
- Item 4 done ahead of time: `sha256sum -c` of every weight file against the block, Qwen3-VL-32B 14 of 14 OK (Slurm job 40381416), Gemma 4 31B 2 of 2 OK (job 40385678), both 2026-11-09.
- No A1 output of any model had been read when the tag was made.

## Limits

- **Same families.** A8 tests scale within Qwen and Gemma, not generality across families.
- **Scale and generation together** for Gemma 4 31B against E4B.
- **Shared keys.** The frozen scripts' outputs carry the sibling's key. The launcher fixes every path, guards the frozen outputs and records provenance.
- **Weights outside the project.** The 32B copy is shared with another project. The checks detect a change; they cannot prevent one.
- **dtype.** bfloat16 only. float32 G2 and G3 are expected to run out of memory, and the frozen script then keeps the bfloat16 verdict. DECISIONS 2026-11-05 accepted that for Qwen3-VL-8B partly because its bfloat16 G2 and G3 were exactly zero. If an A8 model shows a non-zero bfloat16 G2 or G3, no float32 check can tell rounding from leakage, and the paper says so for that model.
- **Cost** estimates assume linear scaling and are upper-end.
- **Environment.** `a8_job.sh` sets up each step's environment as the frozen job does. What differs: `MR_EU_HPC_MODELS_DIR` is set by the launcher from the block, not by the slurm file; the rotation job's `python -m paper_v2.option_score --write_rotation_lookups` is not rerun (the launcher checks the existing lookups against their pins instead); the frozen code runs inside the launcher's Python process (`runpy`), after the launcher has imported `config`, `scripts.evaluate` and the A8 modules; the GPU count may differ from the sibling's (a run setting). None of these enters a computation.

## Dated notes

- 2026-11-09: Gemma 4 31B admitted. The user approved storage of its weights on the lab project store. No output of it exists. Recorded in the block as its status.
- 2026-11-09 (before any A8 output): a per-model step whose output file exists but is incomplete after two out-of-memory attempts at the largest GPU count used for that step counts as "no result" for that model, by the same rule as `not_tested_after_retry`; the other models' analyses continue. Final-check edge case in `research/reviews/a8_audit.md`.
- 2026-11-09 (Freeze checklist item 5, read-only, before any A8 output): Gemma 4 E4B A1 task times from Slurm accounting, arrays 39982133 and 39982134 (real and mismatch; identified by output-file times, since the model was passed by environment variable): 33 min 10 s to 38 min 45 s per rotation task, 10 tasks. Molmo2 (39982135, 39982136): 42 to 49 min. Qwen3-VL-8B (39982137_0, _1): 7 h 44 min and 7 h 42 min. A1 cost is therefore model-specific, not proportional to parameter count; the compute section's Gemma estimate is a run-planning figure only and changes no rule.

**2026-11-09, about 17:48, checklist items 3 and 6 (after the Phase 2 sync, before any A8 output).** On the cluster in `~/rds/hpc-work/study3`, `sha256sum -c` against the block: all 48 entries of `frozen_code_sha256` OK, and all 10 entries of `data_sha256` OK, including the five rotation lookups `data/eu_emotions_human_entropy_rot{0..4}.json`, so the off-repo regeneration and the cluster copies agree. The Phase 2 sync itself logged 113 pinned files with no mismatch (`research/reproduction/phase2_hash_check.json`). Item 2 is still open: the GPU equivalence test, job 40399722, is waiting on priority.

**2026-11-12, about 23:30 (before any A8 output; run logistics, no design change).** The launcher's `a4_analyse` step cannot run on the Mac, because its storage check calls CSD3's `quota`, and it could not have run on the cluster, because the frozen `scripts/a4_analyse.py` reads the Snoek archive at its default `data/external/snoek_au/au_emotion_data.zip`, which was only on the Mac. The archive (CC BY 4.0) was copied to that path on the cluster and made read-only; its sha256 there is `6542ef31...d706`, equal to the pin in `scripts/reproduce.py`. The other inputs `a4_analyse` reads by default (`human_au_effects.json`, `a4_human_ceiling.json`, `a4_recovery.json`) were checked equal by sha256 on the cluster, and `results/a4/manifest.csv` is there. So A8's A4 analysis runs on the cluster through the launcher, as the block's command fixes. The frozen A4 analysis of the two frozen models still runs on the Mac (`post_phase3.py`).

**2026-11-13, about 06:00 (before any A8 output; no design change): A2's and A4's rulings of 2026-11-12 carried over to A8.**

A8 runs A2's and A4's frozen analyses on two larger models (block `steps`: A2 "with both dated deviations and the .01 sensitivity", A4 "PREREG_A4_DRAFT.md"), and its rows are read under the frozen rows. Independent audits of A2's and A4's code produced rulings on 2026-11-12, after A8 was frozen, on points their texts left open (`PREREG_A2_DRAFT.md` and `PREREG_A4_DRAFT.md`, "Deviations", 2026-11-12). A8 takes the same readings, as A9 did for A4's, fixed now:

1. **A2 (rulings 1 to 9) apply to each A8 model's A2 output as to the frozen ones.**
   - The fill rule governs Opposes verdicts (ruling 1).
   - `research/scripts/a2_declared_extras.py` runs on each A8 output. It also writes the equivalence intervals and the shuffles (rulings 4 and 5).
   - Before `a8_analyse_a2.py` runs, an A8 pre-run check applies rulings 2, 3, 6 and 7 to A8's own inputs:
     - the cells and run reports under `results/a8/<a8_id>/stimulus_interventions/`;
     - the A8 model's own delivery record, for pixel shapes;
     - A8's own gate files and A1 and A3 summaries, with their provenance and launcher records;
     - the pinned geometry.
   - The small siblings' gate files are never read for an A8 model.
2. **A4 (rulings 1 to 8) apply to A8's run of A4's analysis.** A8's block settles three points differently:
   - *Family.* The block's family stands: the frozen Holm over the models with A4 cells in `results/a8/a4` (block `families.a4`). The single run starts only when every model with any A4 cell there has all its shards complete, so no model enters the family with part of its data.
   - *Shards.* The shard count is a run setting.
     - For each model it is the `n_shards` of its ok `a4_run` launcher records, which must agree.
     - Each shard file holds exactly `planned_cells(manifest, k, n_shards)`, once each, with status ok and A4's design hash.
     - Each shard has an ok `a4_run` launcher record with the design's weights (block rule `inputs_from_launched_jobs`).
   - *Rows that are not finite (ruling 2).* The block fixes `--cells_dir results/a8/a4`, so the analysis cannot be pointed at a filtered folder.
     - The pre-run check moves each shard file that holds such a row, unchanged, into `results/a8/a4/unfiltered/`. The frozen analysis reads only `cells_<model>_<n>.jsonl` at the top of the folder, so it does not read that subfolder.
     - The check then writes the filtered copy under the original name.
     - The removed rows, and the sha256 of both versions, are recorded with the run.
   - Rulings 4 to 8 apply as written:
     - inputs at their pinned bytes;
     - gates built by `research/scripts/a8_a4_gates.py` from A8's outputs;
     - Inconclusive where Aligned and Not aligned both hold;
     - R_S as the code computes it;
     - single-threaded BLAS (enforced by the launcher since 2026-11-13);
     - the reporting points.
3. **No A4 result for one model.** A model's A4 may be unable to finish, under the dated note of 2026-11-09 (two out-of-memory attempts at the largest GPU count used for a shard). That model then has no A4 result. Its shard files are moved, unchanged, to `results/a8/a4/no_result/` before the single run, so the other models' analysis goes on, and the move is recorded with the run. The model is then outside the block's family, which counts only models with cells in `results/a8/a4`.

Nothing in the block changes; thresholds, families and hashes are as frozen. Recorded in DECISIONS 2026-11-13.
