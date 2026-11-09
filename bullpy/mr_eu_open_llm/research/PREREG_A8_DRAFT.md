# DRAFT amendment A8: larger open-weight models through the frozen pipelines

**Status: DRAFT, design text freeze-ready, not frozen. Nothing here binds until it is audited, hashed and tagged `a8-freeze`. No A8 model has run on any study stimulus. The filename keeps "DRAFT", as the other amendments do, so that references in code stay valid.**

Current `design_sha256` of the block below: `49665a159771043e50e4799d972e3deaa84b1f03a04009bdb458abf6fbea11eb`

The hash is the sha256 of the bytes between the fence lines of "Design block", without the fences, computed by `research/scripts/a8_design.py` in the same way `scripts/a7_design.py` computes A7's. Any edit to the block changes it, and the value pinned in `a8_design.py` must change with it, on purpose.

**Written:** 2026-11-09, while the Qwen3-VL-8B A1 arrays (39982137, 39982138) were still running and before any A1 rotation file was read. Lead decisions on the first draft's open questions were written in the same day (section "Lead decisions, 2026-11-09"). Supporting checks: `research/notes_a8_models.md`.

**What was known when this was drafted.** The A3 result for the three current models (Qwen3-VL-8B accuracy .282 on `single_face`, Gemma 4 E4B .371, Molmo2 .072; `RESULTS_REGISTRY.md`, A3 section), the vision gates (Gemma and Qwen pass G2 to G4, Molmo fails), input delivery, the readout ladder, and A2, A4, A5 and A7 as frozen. No A1 result. The motivation is that the two eligible models are small and weak on the task, so confirmatory claims rest on two weak models. The A8 models were chosen on licence, size, family and storage, before either produced an output on any study stimulus.

**Order of freezing.** The design text and its block are frozen before the Qwen3-VL-8B A1 result is read. Code pins (section "Code hashes") follow as a dated, pre-data step, before any A8 output exists. Shard counts and GPU counts are run settings that cannot change any cell (see "Run settings"); they are set by dated notes after the timing smokes.

**Depends on:** the frozen A1 (`research/PREREGISTRATION.md`, "Amendment A1" and "A1.1"), A2 (`research/PREREG_A2_DRAFT.md`), A3 (`research/PREREG_A3_DRAFT.md`) and A4 (`research/PREREG_A4_DRAFT.md`), unchanged, and the DECISIONS entries cited below.

**What A8 is not.** It does not change any frozen threshold, family, estimand, prompt, readout, item set, code file or verdict. It cannot rescue a frozen verdict and cannot weaken one. Its results are reported beside the frozen ones, in their own rows, under their own multiplicity control.

## Lead decisions, 2026-11-09

On the eight open questions of the first draft, before any A8 output and before any A1 result was read:

1. Qwen3-VL-32B is the confirmed arm. Gemma 4 31B is conditional on the user approving storage on the lab project store. It enters only by a dated note in this file, written before any of its outputs exist. Until then it is not part of A8, and no family includes it.
2. Two models only. InternVL3.5 and Mistral-Small-3.2 are out of A8 and named as future work.
3. A1 on the 32B models runs through the sharded runner, with the partner map of the full item set, checked against the frozen script by an equivalence test on a 6-item smoke. `ampere-long` was checked read-only and is not open to this account (see Compute); it is recorded as a fallback only if access is granted.
4. A8 jobs queue behind the frozen Phase 2 runs.
5. Gemma 4 31B gates are strict: if the frozen `gemma4` gate adapter fails on 31B, there is no gate result and the model is ineligible for A2 and A4.
6. Natural-clip RQ1 runs on A8 models, descriptive only.
7. The design text is frozen before the Qwen3-VL-8B A1 result is read; code pins follow, dated and pre-data.
8. Budget is phased with the stop rules below. Up to about 600 GPU-hours for Qwen3-VL-32B is accepted (free allocation).

## Question

Do the frozen tests give the same answers on larger, newer open-weight models in the same families as the eligible models? A8 asks this of RQ0 (A1 and A3), the vision gates and delivery, Gate B and the stimulus arm (A2), the randomised cue arm (A4), and, descriptively, natural-clip RQ1. Each is judged by its frozen rule.

## Models

Free, open weights, Apache-2.0, not gated, verified on Hugging Face on 2026-11-09 (API `sha` of `main`; details in the notes file).

| A8 id | Status | Hugging Face repo | Pinned revision | Licence | Params | bf16 weights | Frozen code key |
|---|---|---|---|---|---|---|---|
| `qwen3vl_32b` | Confirmed | `Qwen/Qwen3-VL-32B-Instruct` | `0cfaf48183f594c314753d30a4c4974bc75f3ccb` | Apache-2.0 | 33.36 B | 66.7 GB | `qwen3vl` |
| `gemma4_31b` | Conditional on storage approval | `google/gemma-4-31B-it` | `842da3794eaa0b77d5f08bae87a17459d91ff475` | Apache-2.0 (the card's `license_link` serves the Apache 2.0 text) | 31.27 B | 62.5 GB | `gemma4` |

**Why these.** Each frozen pipeline script whitelists the keys `qwen3vl`, `gemma4` and `molmo2` and branches on them. A new architecture needs a new key, which means editing a frozen file or swapping in new branches at run time. Qwen3-VL-32B has the same class (`Qwen3VLForConditionalGeneration`) and the same vision tower as Qwen3-VL-8B. Gemma 4 31B has the same class (`Gemma4ForConditionalGeneration`) as Gemma 4 E4B. Both load through `scripts/evaluate.resolve_model_path`, which reads `MR_EU_HPC_MODELS_DIR`, so they run under the existing keys, with a different weights directory and different output paths and no code edit. Each is a scale-up of an eligible model, so a difference from its small sibling is read as scale within a family. Gemma 4 31B also differs from E4B in more than size (no KV sharing, no per-layer inputs, no audio tower).

**Future work, not in A8.** A third family (InternVL3.5-38B-HF, revision `7c830fc2...`, whose language model is Qwen3; Mistral-Small-3.2-24B-Instruct-2506, revision `95a6d26c...`, which ships no video preprocessor config). Either needs new model code in every pipeline and its own amendment.

**Not proposed.** Gemma 3 27B (Gemma Terms of Use, gated), Gemma 4 26B-A4B (mixture of experts), Qwen3.5-27B (no key in the code), Cosmos-Reason2-32B (a task-tuned fine-tune of the same architecture), Llama 4 Scout (gated, too large for one A100), larger Molmo2 checkpoints (Molmo2 fails G2 and G4).

## Storage

**Constraint.** RDS hpc-work is at 1041.9 GB of a 1099.5 GB soft limit (1209.5 hard) and 996,131 of 1,048,576 files (`quota`, 2026-11-09). ROADMAP Part 3 stops new writes above 1,080 GB or 1,040,000 files. No 30 B class model fits as a new download on RDS. `/home` is at 49.8 of 52.4 GB. Slurm reports `TMP_DISK` 0 on ampere nodes.

**Qwen3-VL-32B: read in place, no new bytes on RDS.** The copy at `/rds/project/rds-ePtR33Nsgi4/eb2007/models/Qwen3-VL-32B-Instruct` is in the user's own directory of the lab project store, which has its own quota (14,341 of 23,000 GB; 7.70 M of 11.78 M files). It is owned by `eb2007` and readable by our jobs. Its download metadata name revision `0cfaf481...`, its shard sizes match Hugging Face, and the stored sha256 of shards 1 and 14 match the Hugging Face LFS hashes. STATE.md says CASCADE loads this copy, so A8 only reads it. The one write A8 needs is a directory holding one symlink, `~/rds/hpc-work/study3/a8_models/qwen3vl -> /rds/project/rds-ePtR33Nsgi4/eb2007/models/Qwen3-VL-32B-Instruct`, so that `MR_EU_HPC_MODELS_DIR` resolves the key `qwen3vl` to the 32B.

**Gemma 4 31B, if approved.** A single download of the pinned revision into `/rds/project/rds-ePtR33Nsgi4/eb2007/models/gemma-4-31B-it` (12 files, 62.5 GB), with the Hugging Face cache pointed at the project store and not at `study3/hf_cache` on RDS, then the same symlink pattern under `~/rds/hpc-work/study3/a8_models_gemma/gemma4`.

**Integrity.** Every weight file's name, size and Hugging Face LFS sha256 at the pinned revision is in the design block. Before the first A8 job, the coordinator runs `research/scripts/a8_preflight.py --full_hash` read-only on the cluster and records the result; any file whose sha256 differs stops A8 for that model. Every A8 job then runs the preflight without `--full_hash`, which checks names, sizes and the resolved path, and stops on any difference. A full hash runs again after the last A8 job, and the result is recorded.

**Outputs.** Cells stay JSON lines, never one file per item. All A8 output goes under `results/a8/` and `research/reproduction/a8/`. The preflight refuses any other output path. No A8 job starts above the storage stop rule.

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
      "expected_config": {"architectures": ["Qwen3VLForConditionalGeneration"], "model_type": "qwen3_vl", "text_num_hidden_layers": 64, "text_hidden_size": 5120},
      "a1_sampling": "sequential",
      "weight_files": {
        "model-00001-of-00014.safetensors": {"size": 4932328944, "sha256": "6b9dfbc930e505402ae9d7e5091a9d7d656cda5f34614f01cfe70bfb0cca27cb"},
        "model-00002-of-00014.safetensors": {"size": 4875990528, "sha256": "d8bb44b4ff303fe76fe9e894022fb3dc71b15a2e716592790fe0e3c3e60478fa"},
        "model-00003-of-00014.safetensors": {"size": 4875990552, "sha256": "54f22e8b3168f8dc962fac0d313607ebf52a12b433d4cf3098a0d82d9f042940"},
        "model-00004-of-00014.safetensors": {"size": 4875990584, "sha256": "ad09c74d3c13ee29b5d0d84548fd8a3424a651564eaccd519946c296e59c557f"},
        "model-00005-of-00014.safetensors": {"size": 4875990584, "sha256": "fc993c8a0e2a5b0570f383e1a95dc3a1281d1b224b6f3ee908f4827941e1dfc2"},
        "model-00006-of-00014.safetensors": {"size": 4875990584, "sha256": "82f05620d1f718a90c362b221d6a184ff1a0f53301d706882d3df49695fa1974"},
        "model-00007-of-00014.safetensors": {"size": 4875990584, "sha256": "fb91da8cb01ff4de3eef0eab1c3e769a734b3a1aafc61734068638a0d6c86934"},
        "model-00008-of-00014.safetensors": {"size": 4875990584, "sha256": "431ca56535c8781944ce3801f5eb61c45531e853ecc5846d936ebaf4761b764f"},
        "model-00009-of-00014.safetensors": {"size": 4875990584, "sha256": "3825e3f4302f4d2f7d76aa7430d2ce0864fde6b9e540a5806bf0d8e38e4d9f47"},
        "model-00010-of-00014.safetensors": {"size": 4875990584, "sha256": "aded5a4d1d5e22dbd8b6f79266b6eb88c840411b09527c53917a1419ace22e2f"},
        "model-00011-of-00014.safetensors": {"size": 4875990584, "sha256": "3820ffe8d8d6477f6fe8d614ef3c87abb264ee39accebf43a1507b970d80946f"},
        "model-00012-of-00014.safetensors": {"size": 4875990584, "sha256": "05ad2d08ce71963121c9b03f1d9ec5d7641052f4b23c6c12b80d71065eb8e98e"},
        "model-00013-of-00014.safetensors": {"size": 4875990584, "sha256": "b64f2289871261fdd1abbd3b78bcd66011b341de3dc8eeb2ed1a473ee7c8d95c"},
        "model-00014-of-00014.safetensors": {"size": 3270697008, "sha256": "e45b6c9998c77ee5a6577f9f47bc76416c1d4d387169e50c4c9d3134ea51b13b"}
      }
    },
    "gemma4_31b": {
      "status": "conditional_on_project_store_approval",
      "repo": "google/gemma-4-31B-it",
      "revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
      "licence": "apache-2.0",
      "code_key": "gemma4",
      "weights_dir": "/rds/project/rds-ePtR33Nsgi4/eb2007/models/gemma-4-31B-it",
      "models_dir": "~/rds/hpc-work/study3/a8_models_gemma",
      "expected_config": {"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4", "text_num_hidden_layers": 60, "text_hidden_size": 5376},
      "a1_sampling": "batched",
      "weight_files": {
        "model-00001-of-00002.safetensors": {"size": 49784788364, "sha256": "eeef8791537bc04f110967c513149e037d2a9ae97d49add7291ebfa62806bbfa"},
        "model-00002-of-00002.safetensors": {"size": 12761549884, "sha256": "018912220f559f7025d60333e0996183cd538aa77ad6f4988a89ce47be681f10"}
      }
    }
  },
  "output_roots": ["results/a8", "research/reproduction/a8"],
  "storage_stop": {"rds_gb": 1080, "rds_files": 1040000},
  "dtype": "bfloat16",
  "frozen_code_sha256": {
    "scripts/evaluate.py": "cc672f9438fe0b165dc2d620f5fad0e0c3a8845857a91e68871e0dad669a2481",
    "scripts/model_inference.py": "2ddb946f75db80de14299a77b0de682d1b16ce092185f64d3b3e2d7ae0e7fc7c",
    "scripts/multi_frame.py": "2cdc5c454f0fe9ecccbd167d7b145d29b7a9d602bbb7a6e5cf61c28cbe4e1b35",
    "scripts/frame_sampling.py": "11b1bb42f8565e66ff0be457bd76e4649410a8991cfe841280eba2444d90becb",
    "scripts/forced_choice_entropy.py": "1578a7242f7217aaab6c2bcdaef8ce82df0261c05a249960df3646a6d1f2799f",
    "scripts/prompts.py": "871272d4aa6c86ff809f7631a4db81aeeb41564ba7f0a31809ea08e6dcf64ba7",
    "scripts/trial_foils.py": "5e5d23a085ee4465d983e3bf12f7de48d95b0a2fd16287b6529bdf3c093d9ebf",
    "scripts/tolerant_parse.py": "d4e31b4f94eb28480edb1f2a9e41ed12a6f83530dd5545881bd56d0fcfa58833",
    "scripts/emotion_parse.py": "19a8f875da09bdb4b7ba679eb90ee72adbb37c31c8f2b4e74e2c9e4d445ba93d",
    "src/paper_v2/option_score.py": "ae949e4c1c3abc6a775a535c4e0a5042bce59ba14f7ac1afeefd476f032a5a2b",
    "src/paper_v2/schema.py": "4acc626669439384115ae0efcb52cec4edee02c07662a242212d5f7b9c359b47",
    "config.py": "75d4a11128aad7d026cbc66331ed9cfd0c51acee1b64a986c2b2138cd1ba77e8",
    "research/scripts/decoder_preflight.py": "ed5262d89562e2662f6aea3660edb4f1d9cb5306a08ffc68daddd4f669c20699",
    "research/scripts/analyse_rotation.py": "98890c5bff65f13abeb0549b61fe414dc9fa30082b84bcb564671683ec12e24f",
    "research/scripts/analyse_rotation_run.py": "7f52f9e7b29f0afbbe76977ee3b3330f01c8f6ececf29fe14ddf9655c67d653e",
    "research/scripts/same_clip_family_analysis.py": "2e633c534bf1bf0fbfd6d133aff587003be30578ee09ecf800d4eedce5de493f",
    "scripts/face_input_validity.py": "8a681a652d8e34065372d4f45bce650ba5276b5138e3aa64cf4310d65e93d818",
    "scripts/vision_gates.py": "6327b6f88e99409cba5c63d5aca0fe97348fbce4f3b08522158a8d9ba3ce8194",
    "scripts/readout_ladder.py": "738b2c5031e5f7f1c0337896b0e8e680b2f767d0f33c850588f743d6c5595c1c",
    "scripts/stimulus_design.py": "89d256c1072e0768fc747294553c5710adaf42184a175866d7bd75bcc6104978",
    "scripts/stimulus_interventions.py": "93c601eb42d6700d4f0fc8ba72317ed5c2e354105d57fcac611c59660125deb3",
    "scripts/run_stimulus_interventions.py": "f7880c94e9671cc294cd4eaeece2108d7cd1d7a4c7e6f9e30a1c495964e6cd82",
    "scripts/analyse_stimulus_interventions.py": "aba4f533daad6210e87f7bb4e68fc4627805cb4dfd9d9308df734fada4edc8b1",
    "scripts/a4_design.py": "a0bf7bb1c69ec7af1f11fabb5a9f31a6c694a30b92e624f1e1c241f161d062a8",
    "scripts/a4_run.py": "f842e574bf5cdaebbe9057d0ecc0e311c900dd54ff0ca221d1f9baa3768a8dc6",
    "scripts/a4_analyse.py": "30f8f529d128b555ccfbf4d31ac3de70d91f90b1f5bfd52360ae9f39305fe56d",
    "slurm_jobs/study3_rotation_eval.sh": "59fe95123cc83b5710bb47de824c608cf4f6fac8ebd864d383e77eefe71e6230",
    "slurm_jobs/study3_face_input_validity.sh": "2bf3a9f74e6d6a2929e34fc54249e79ea9216b50460f03c89c1a7d3e058d0d8a",
    "slurm_jobs/study3_vision_gates.sh": "3607c0407319d9b4fdd333102e4776765e510470874cd243301ba6a11c97b705",
    "slurm_jobs/study3_stimulus_interventions.sh": "1c8d22ea637abedd3a3c2332331639c7e33a7e7ed9cb2e741335d17ce138eec5",
    "slurm_jobs/study3_readout_ladder.sh": "a4ed37971dc28348f9d67052804b72a530c1cf59537d5116d60f7d04be9f39bd",
    "slurm_jobs/study3_a4.sh": "dd9a360b20701742fd70d849b92c964d6b992c966ded744fa4125e0166103c26"
  },
  "frozen_a3_summary_sha256": "c1c64e6975b32356a3314268c054f0dee483ddf74b11cd66d03b2faf972e64a4",
  "frozen_design_sha256": {"A2": "0ae22118d3b2690c581595890b9bb8b904e1f7860f37be9ac909f925b7058de7", "A4": "12c454da512bfe4c239e6c3ef3610d80df6fd7c2e9119a9ff8df913a3c1a3e94"},
  "steps": [
    {"step": "vision_gates", "source": "DECISIONS 2026-11-04 and 2026-11-05; scripts/vision_gates.py GATE_RULES", "output": "research/reproduction/a8/{a8_id}/vision_gates_{key}.json"},
    {"step": "input_delivery", "source": "PREREG_A2_DRAFT.md, Delivery checks", "output": "research/reproduction/a8/{a8_id}/input_delivery_{key}.json"},
    {"step": "readout_ladder_descriptive", "source": "PREREG_A2_DRAFT.md, Readouts; the readout is not re-chosen", "output": "research/reproduction/a8/{a8_id}/readout_ladder_{key}.json"},
    {"step": "a3", "source": "PREREG_A3_DRAFT.md", "output": "results/a8/{a8_id}/face_input_validity/cells_{key}.jsonl"},
    {"step": "a1", "source": "PREREGISTRATION.md, Amendment A1 and A1.1; DECISIONS 2026-11-08, A1 errored trials", "output": "results/a8/{a8_id}/rotation/eval_v2_eu_emotions_{a8_id}_video_only_seed42_rot{r}_{c}.json"},
    {"step": "natural_clip_rq1_descriptive", "source": "DECISIONS 2026-11-07, Natural-clip RQ1 under each RQ0 outcome", "output": "research/reproduction/a8/{a8_id}/"},
    {"step": "a2", "source": "PREREG_A2_DRAFT.md, with both dated deviations and the .01 sensitivity", "output": "results/a8/{a8_id}/stimulus_interventions/"},
    {"step": "a4", "source": "PREREG_A4_DRAFT.md", "output": "results/a8/a4/"}
  ],
  "a1": {
    "rotations": [0, 1, 2, 3, 4],
    "conditions": ["real", "mismatch"],
    "evaluate_args": {"dataset": "eu_emotions", "condition": "video_only", "stage": "stage2", "max_frames": 16, "fps": 1, "seed": 42, "n_options": 6, "data_root": "data/eu_emotions", "manifest": "data/eu_emotions_full_manifest.json", "human_options": "data/eu_emotions_human_entropy_rot{r}.json", "fc_samples": 20, "vision_mismatch_seed": 42},
    "sharding": "trials assigned by manifest position modulo n_shards; every shard loads the full manifest, so the frozen code computes mismatch partners over all items; only the scoring loop is restricted",
    "merge": "one file per rotation and condition, trials in manifest order, exact partition of the items required"
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
    "a1": "Holm over A8 models that run A1",
    "a3": "Holm over A8 models that run A3",
    "a2": "Westfall-Young step-down maxT within one model and route, as frozen",
    "a4": "Holm over A8 models eligible for A4, within a dataset",
    "pooling": "none, with frozen models, with each other, or with small siblings"
  },
  "frozen_verdicts_unchanged": true,
  "natural_clip_rq1": "descriptive only",
  "queue": "after both Qwen A1 arrays complete, all 30 rotation files exist, A8 is frozen, and the frozen Phase 2 runs are submitted",
  "equivalence_tests": {
    "a1_sharding": "on Qwen3-VL-8B, frozen evaluate.py with --max_trials 6 against the shard runner on the same 6 items in 2 shards, merged; every trial record identical",
    "a3_summary": "the A8 summary code on the three frozen cell files reproduces frozen_a3_summary_sha256 byte for byte; on synthetic cells it equals the frozen summarise output",
    "a2_loader": "with the frozen A3 file as the A8 A3 file, the A2 wrapper's output equals the frozen analysis output byte for byte on synthetic cells"
  }
}
```

## Pipelines each model runs, unchanged

Every step uses the frozen script and constants. The A8 slurm files are copies of the frozen slurm logic with only paths, wall time, GPU count, the preflight call and `HF_HOME` changed. Each step's output path is explicit, because under the shared keys a default path would overwrite a frozen output of the small model.

| Step | Frozen source | Frozen code run | A8 changes |
|---|---|---|---|
| 1. Vision gates G2 to G4 | DECISIONS 2026-11-04 and 2026-11-05; `scripts/vision_gates.py` `GATE_RULES` | `python -m scripts.vision_gates --model <key> --output research/reproduction/a8/<id>/vision_gates_<key>.json` | Wall time 12 h. float32 is expected to run out of memory; the bfloat16 verdict then stands, by the DECISIONS 2026-11-05 ruling. Gemma 4 31B: strict (decision 5) |
| 2. Input delivery | `PREREG_A2_DRAFT.md`, "Delivery checks" | `run_stimulus_interventions --step delivery --model <key> --reproduction research/reproduction/a8/<id> --output_dir results/a8/<id>/stimulus_interventions` | None |
| 3. Readout ladder, descriptive | `PREREG_A2_DRAFT.md`, "Readouts" | `readout_ladder --step ladder --model <key> --output research/reproduction/a8/<id>` | The readout is not re-chosen. Coverage on cells is judged by Gate B2 |
| 4. A3 | `PREREG_A3_DRAFT.md` | `face_input_validity --model <key> --output_dir results/a8/<id>/face_input_validity`, with `A3_FROZEN=1` | Longer wall time (the script resumes). Summary by `a8_face_input_validity_summary.py` |
| 5. A1 | `PREREGISTRATION.md`, A1 and A1.1; DECISIONS 2026-11-08 | `scripts/evaluate.py`, with the frozen rotation script's arguments, through `a8_rotation_shard.py` | Sharded and merged (Compute). Sampling by the frozen script's rule by key: sequential for `qwen3vl`, batched for `gemma4` |
| 6. Natural-clip RQ1, descriptive | DECISIONS 2026-11-07 | `same_clip_family_analysis.py --model-route` on the merged A8 real-condition files | Read by the DECISIONS 2026-11-07 table, with every row descriptive for A8 |
| 7. A2, if eligible | `PREREG_A2_DRAFT.md` | `run_stimulus_interventions --step run`, then `a8_analyse_a2.py` | Shard count is a run setting |
| 8. A4, if eligible | `PREREG_A4_DRAFT.md` | `a4_run.py --a4_frozen --output_dir results/a8/a4`, then `a4_analyse.py --cells_dir results/a8/a4 --gates research/reproduction/a8/a4_gates.json` | Shard count is a run setting. With one A8 model per key in that folder, the frozen Holm over `d.MODELS` is the A8 family (to be confirmed by a dry run on synthetic cells before the freeze) |

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

A8 forms its own families, of A8 models only. The frozen families are untouched: A1 Holm over Qwen3-VL-8B, Gemma 4 E4B and Molmo2; A3 Holm over the same three; A4 Holm over Gemma 4 E4B and Qwen3-VL-8B within a dataset. A2 is already per model and route.

With Qwen3-VL-32B alone, each A8 Holm family has one member and Holm is the identity, as in the frozen code with one model. If Gemma 4 31B enters by its dated note, the families have two members. Holm controls familywise error at .05 at any size; a smaller family is less strict per model than the frozen three-model family, and the paper says so.

**No pooling** with frozen models, with each other, or with the small siblings.

**Reporting.** A8 rows are labelled A8 and placed under the frozen rows. A8 results are reported whatever they show. Where an A8 model and its sibling disagree, both verdicts are reported and the difference is described, not tested. A8 may support a scope statement ("the result held, or did not, in a larger model of the same family"), never a change to a frozen verdict.

**Cannot rescue.** No frozen verdict is changed by A8. A frozen Inconclusive stays Inconclusive, a frozen null stays a null, and Molmo2 stays ineligible. If an A8 model passes where its sibling failed, the sibling's verdict stands.

**What is read.** Each A8 analysis runs once, after all its outputs exist, from frozen code and the pinned A8 wrappers. Mechanics and timing smokes are read for status, seconds, memory and forwards only.

## Compute

GPU: NVIDIA A100-SXM4-80GB on `ampere` (`gpu:nvidia_a100:4` per node; GPU name stored in `vision_gates_*.json`). Account `baron-cohen-sl3-gpu` is associated with partition `ampere` and QOS `gpu2,intr` only (`sacctmgr show assoc`, 2026-11-09); `gpu2` has a 12 h wall limit. `ampere-long` allows only QOS `gpul` (`scontrol show partition ampere-long`), which this account does not have, so it is not available. It stays a fallback only if HPC support grants access.

**Assumptions.** Time scales with language-model parameters: 33.36 / 8.77 = 3.80 for Qwen3-VL-32B over 8B (Gemma 4 31B over E4B: 3.9 by stored parameters, up to about 7.8 if E4B computes as about 4 B, UNVERIFIED). This is upper-end, since the vision tower is the same size on both Qwen models and decoding and data loading do not scale. One GPU per job; a step that needs two (`--gres=gpu:2`) doubles its GPU-hours.

| Step | Base, measured | Qwen3-VL-32B, GPU-hours | Gemma 4 31B, if it enters |
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

Qwen3-VL-32B is within the accepted 600 GPU-hours at the central estimate and about at it at the upper end. A8 runs in phases, each started only when the last allows it: Phase A, then A1 if the stop rule allows, then A2 and A4 if eligible. If a measured phase projects Qwen3-VL-32B over 600 GPU-hours in total, A4 is the step deferred, by a dated note before it runs. GPU-hours available: 2,859 (STATE.md item 1, 2026-11-09).

**A1 sharding.** `scripts/evaluate.py` cannot resume, and a 32B task projects to about 29 h against the 12 h limit. Splitting the manifest is not allowed, because `evaluate.py` computes mismatch partners over the trials it loads, and a mismatch trial reads its partner's video from the same loaded set. `research/scripts/a8_rotation_shard.py` therefore runs the frozen `evaluate.main()` with the full manifest, so the frozen `vision_mismatch_partners` builds the partner map over all 243 items exactly as in the unsharded run. Only the scoring loop is restricted to the shard's items, by giving `scripts.evaluate` a module-level `enumerate` that yields the shard's `(index, trial)` pairs, with their original manifest indices, when it is applied to the loaded trial list, and behaves as the built-in otherwise. Draws are seeded per trial (`seed_generation(seed, trial_id, k)`), and the option set uses the original index, so a trial's record does not depend on its shard. The runner also computes the full-set partner map itself with the frozen function and checks every mismatch record's `vision_source_trial_id` against it. Shards are merged into one file per rotation and condition, in manifest order, and the merge refuses a gap, a duplicate or a mixed run. The equivalence test is in the design block. Four shards per task put a 32B shard at about 7.4 h.

**Run settings.** Shard counts (A1, A2, A4) and GPU counts are set by dated notes after the timing smokes, before the run. They cannot change any cell: A1 by the equivalence test, A2 and A4 because their frozen analyses accept any complete set of shards.

**Timing.** No A8 job is submitted until both Qwen A1 arrays show COMPLETED for every task, all 30 rotation files exist, A8 is frozen, and the frozen Phase 2 runs (Gate B and A2 on the eligible models, the A4 timing smoke and runs) are submitted.

## Code

No existing file is edited. New files, written 2026-11-09 and to be pinned in "Code hashes":

| File | Purpose |
|---|---|
| `research/scripts/a8_design.py` | Reads the design block, checks its hash against the pinned and stated values, checks every frozen code file against `frozen_code_sha256`, and checks the A8 code pins once they exist |
| `research/scripts/a8_preflight.py` | Run first in every A8 job: resolves the key through the models directory as `resolve_model_path` does, requires it to be the design's weights directory, checks `config.json` and every weight file's name and size (and sha256 with `--full_hash`), refuses outputs outside the A8 roots or over an existing file, applies the storage stop rule, and writes a provenance JSON |
| `research/scripts/a8_rotation_shard.py` | A1 shard runner, merge and partner check |
| `research/scripts/a8_analyse_rotation.py` | A1 analysis over the A8 family: the frozen `analyse_rotation` through the errored-trial wrapper, with its input template pointed at the merged A8 files and its model list set to the A8 ids |
| `research/scripts/a8_face_input_validity_summary.py` | A3 summary over the A8 family: frozen `read_cells`, `summarise_model` and `decide` |
| `research/scripts/a8_analyse_a2.py` | A2 analysis: the frozen `analyse_stimulus_interventions.main`, with only `a3_summary` replaced by a loader of the A8 A3 summary whose sha256 is given and checked |
| `tests/test_a8.py` | Unit tests and the three equivalence tests; GPU and real-model tests skip without them |

Still to write before any A8 job: the A8 slurm files (each diffed against its frozen original, diff pinned here), the A4 gates writer, and relabelling for tables.

## Code hashes

```text
pending: pinned by a dated, pre-data step before any A8 output exists
```

## Recovery

Not needed. Recovery in A2 and A4 checks that the analysis recovers known observers on the real items, actors, contrasts and constants; it does not depend on which model produced the cells. A8 uses the same analysis code, items, actors, contrasts, constants and design hashes, so the frozen recovery results apply. A1 and A3 had no recovery step. What is new is checked by equivalence tests instead: sharding (identical records on a 6-item smoke), the A3 model-set wrapper (identical bytes on frozen and synthetic inputs), and the A2 loader swap (identical bytes on synthetic cells). Holm on a smaller family needs no recovery. Recovery would be needed only if an analysis constant, contrast, test or family structure changed beyond membership; none does.

## Freeze checklist

1. Design block audited; hash stated above and pinned in `a8_design.py`; tag `a8-freeze` on the design text, before the Qwen3-VL-8B A1 result is read; DECISIONS entry.
2. Dated pre-data step, before any A8 output: code written and tested, equivalence tests run (the A1 one on the cluster), "Code hashes" filled, slurm diffs pinned, timing smokes read for mechanics only, run settings set.
3. `a8_preflight.py --full_hash` run read-only on the 32B copy; every sha256 equal to the block.
4. Gemma 4 E4B A1 task time read from Slurm accounting, if Gemma 4 31B enters.
5. Frozen code on the cluster equal to `frozen_code_sha256` after the Phase 2 sync. On 2026-11-09 every file matched except `scripts/run_stimulus_interventions.py` (cluster `19b1fdae...`, pre-sync, expected per ROADMAP 2.1).
6. Storage check at submission.

## Limits

- **Same families.** A8 tests scale within Qwen (and Gemma, if it enters), not generality across families.
- **Scale and generation together** for Gemma 4 31B against E4B.
- **Shared keys.** Outputs of the frozen scripts carry the small sibling's key. The preflight refuses frozen paths and records provenance; a mistake there would mix models.
- **Weights outside the project.** The 32B copy is shared with another project. The checks detect a change; they cannot prevent one.
- **dtype.** bfloat16 only; float32 G2 and G3 checks are expected to run out of memory.
- **Cost** estimates assume linear scaling and are upper-end.

## Dated notes

- 2026-11-09: admit `gemma4_31b`. Reason: the user approved the 62.5 GB download of the pinned revision into the lab project store (`/rds/project/rds-ePtR33Nsgi4/eb2007/models/gemma-4-31B-it`) on 2026-11-09. None of its outputs exist; the download had not started when this line was written.

Other notes: none yet. Gemma 4 31B's entry, run settings and code pins go here, each dated and stated as written before the outputs it concerns. A conditional model enters only by a line of the form "- YYYY-MM-DD: admit `gemma4_31b`", followed by the reason and the statement that none of its outputs exist; `research/scripts/a8_design.py` reads that line, and every A8 family is the confirmed models plus the admitted ones.
