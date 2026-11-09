# Notes for A8: model and storage checks

**Written:** 2026-11-09, by an agent drafting `PREREG_A8_DRAFT.md`. Working notes, not a preregistration. Every value below was read on 2026-11-09 from the source named beside it. Nothing was downloaded, copied, written or submitted on the cluster.

## Hugging Face metadata (API `https://huggingface.co/api/models/<id>?blobs=true`, read 2026-11-09)

| Repo | Revision (`sha`, main) | Licence (card metadata) | Gated | `architectures` in config.json at that revision | Safetensors bytes | Params (HF `safetensors.total`) |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3-VL-32B-Instruct` | `0cfaf48183f594c314753d30a4c4974bc75f3ccb` | apache-2.0 | no | `Qwen3VLForConditionalGeneration` (model_type `qwen3_vl`) | 66.7 GB in 14 shards | 33,357,390,064 |
| `Qwen/Qwen3-VL-8B-Instruct` (current study model) | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | apache-2.0 | no | `Qwen3VLForConditionalGeneration` | 17.5 GB | 8,767,123,696 |
| `google/gemma-4-31B-it` | `842da3794eaa0b77d5f08bae87a17459d91ff475` | apache-2.0; `license_link` https://ai.google.dev/gemma/docs/gemma_4_license, which served the Apache License 2.0 text on 2026-11-09 | no | `Gemma4ForConditionalGeneration` (model_type `gemma4`) | 62.5 GB in 2 shards | 31,273,088,876 |
| `google/gemma-4-E4B-it` (current study model) | `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c` | apache-2.0 | no | `Gemma4ForConditionalGeneration` | 16.0 GB | 7,996,156,490 |
| `google/gemma-4-26B-A4B-it` (not proposed; MoE) | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` | apache-2.0 | no | not fetched | 51.6 GB | 25,805,936,206 |
| `OpenGVLab/InternVL3_5-38B-HF` | `7c830fc25e874f9a861087e3c70587f2b9b555f3` | apache-2.0 | no | `InternVLForConditionalGeneration`; text backbone model_type `qwen3`, vision `internvl_vision` | 76.8 GB in 16 shards | 38,390,405,504 |
| `OpenGVLab/InternVL3_5-14B-HF` | `226b96d5912e69159abc0384cefcbd51487fdce0` | apache-2.0 | no | not fetched | not fetched | not fetched |
| `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | `95a6d26c4bfb886c58daf9d3f7332c857cb27b43` | apache-2.0 | no | `Mistral3ForConditionalGeneration`; text `mistral`, vision `pixtral` | 48.0 GB in HF shards (plus a 48.0 GB `consolidated.safetensors` not needed) | 24,011,361,280 |
| `google/gemma-3-27b-it` (not proposed) | `005ad3404e59d6023443cb575daa05336842228a` | `gemma` (Gemma Terms of Use) | manual | not fetched | 54.9 GB | 27,432,406,640 |
| `Qwen/Qwen3.5-27B` (not proposed) | `fc05daec18b0a78c049392ed2e771dde82bdf654` | apache-2.0 | no | not fetched | 55.6 GB | 27,781,427,952 |

Licence notes: Apache-2.0 allows research use, modification and redistribution with notice; it has no use-restriction policy. Gemma 3 is under the separate Gemma Terms of Use and is gated, which is one reason it is not proposed. The licence of the model, not of its outputs, is what is recorded here; nothing in A8 redistributes weights.

### Config facts that matter for the frozen code

- Qwen3-VL-32B and Qwen3-VL-8B have the same vision tower (depth 27, hidden 1152, deepstack indexes 8, 16, 24). The text model is 64 layers by 5120 against 36 by 4096. Every `qwen3vl` branch in the frozen scripts addresses modules by name (`model.model.visual`, `language_model.layers`, `rope_deltas`, `deepstack_features`), so the 32B should run on them as is. Inferred from reading the code, not run.
- Gemma 4 31B and Gemma 4 E4B share the class, but differ where the frozen gate adapter cares: `text_config.hidden_size_per_layer_input` is 0 on 31B and 256 on E4B; `num_kv_shared_layers` is 0 on 31B and 18 on E4B; there is no `audio_config` content on 31B. `scripts/vision_gates.py` builds the `gemma4` adapter with `has_per_layer=True` and reads `per_layer[0]` from the decoder call (lines 535 to 545). If 31B passes no per-layer input, that line fails. UNVERIFIED until a mechanics run. The Gemma rule "never patch image positions at L24 or later" (STATE item 6) comes from E4B's KV sharing and does not carry to 31B, which shares none.
- `vision_soft_tokens_per_image` is 280 on both Gemma models, matching `VIDEO_KWARGS["gemma4"]` in `scripts/run_stimulus_interventions.py`.

## CASCADE copy of Qwen3-VL-32B (read-only checks on CSD3, 2026-11-09)

- Path: `/rds/project/rds-ePtR33Nsgi4/eb2007/models/Qwen3-VL-32B-Instruct` (the user's own directory in the lab project store `rds-ePtR33Nsgi4`, reached from `~/rds/rds-autism-research-ePtR33Nsgi4`).
- Owner `eb2007`, group `rds-ePtR33Nsgi4-managers`. Directory `drwxrwsr-x`, files `-rw-rw-r--`, so readable by our jobs, which run as `eb2007`. Also writable by the managers group, so it can change under us (see the draft's integrity check).
- `du -sh`: 63G. 14 shards, `model-00001-of-00014.safetensors` to `model-00014-of-00014.safetensors`, sizes identical to the Hugging Face listing at revision `0cfaf481`.
- `.cache/huggingface/download/*.metadata` names commit `0cfaf48183f594c314753d30a4c4974bc75f3ccb` for `config.json`, shard 1 and shard 14. The stored sha256 of shard 1 (`6b9dfbc9...cca27cb`) and shard 14 (`e45b6c99...ea51b13b`) equal the LFS sha256 Hugging Face lists for those files. Shards 2 to 13 were not checked (their metadata were not read); the full hash check is a freeze item.
- `config.json` on disk: `Qwen3VLForConditionalGeneration`, 64 text layers, hidden 5120, written by transformers 4.57.0.dev0.
- Other directories there: `Cosmos-Reason2-32B` (an NVIDIA fine-tune of the same architecture, same family, not proposed), `Qwen3.5-9B`, `Qwen-Drive-1.0-4B`, `Alpamayo2-Super`, `Cosmos-Reason2-8B`, `vjepa2-vitl-fpc64-256`, and a symlink `Qwen3-VL-8B-Instruct -> /rds/user/eb2007/hpc-work/cascade/weights/Qwen3-VL-8B-Instruct`.
- That CASCADE loads its copy from here is stated in STATE.md (2026-11-07). CASCADE code was not read to confirm it.

## Storage on CSD3 (2026-11-09, `quota`, `df -h`)

| Store | Used | Soft / hard | Files used / limit |
|---|---|---|---|
| `/rds-d7/user/eb2007` (RDS hpc-work) | 1041.9 GB | 1099.5 / 1209.5 GB | 996,131 / 1,048,576 |
| `rds-ePtR33Nsgi4` (lab project store) | 14,341.0 GB | 23,000.0 / 23,000.0 GB | 7,698,775 / 11,776,000 |
| `/home` | 49.8 GB | 52.4 / 57.7 GB | 207,498 / none |

`df -h`: RDS user 54G available; project store 7.9T available; `/tmp` on the login node is a 16 GB tmpfs. `sinfo` reports `TMP_DISK` 0 on every ampere node, so node-local staging space is not advertised to Slurm. Whether GPU nodes have a usable local disk was not checked and is UNVERIFIED.

ROADMAP Part 3 stops new writes on RDS above 1,080 GB or 1,040,000 files, which leaves about 38 GB and about 43,900 files.

`study2/models` holds `internvl2` 16G, `llavanext` 16G, `qwen2vl` 16G, `qwen3vl` 17G, `gemma4` 15G, `molmo2` 29G (`du -sh`). COMPUTE.md says not to touch study2, so none of this is proposed for deletion.

## GPU (CSD3)

- `sinfo -p ampere`: nodes with `gpu:nvidia_a100:4`, 1,024,000 MB RAM, 128 CPUs. An `ampere-long` partition with no time limit exists; whether account `BARON-COHEN-SL3-GPU` may use it is UNVERIFIED.
- GPU model on the gate jobs: `NVIDIA A100-SXM4-80GB` (`research/reproduction/vision_gates_{qwen3vl,gemma4}.json`, `stages.*.gpu`).
- Wall limit used by every study3 GPU script is 12 h or less; COMPUTE.md gives a 12 h cap for the SL3 GPU QOS.

## Queue at the time of writing (`squeue -u eb2007`, 2026-11-09)

39982137: tasks 2 and 3 running, 4 pending. 39982138: tasks 0 and 2 running (task 0 at 7 h 49 min), 3 and 4 pending. No other jobs of this user were listed.

## Measured timings used for scaling (this repo)

| What | Model | Time | Source |
|---|---|---|---|
| A1 rotation task (243 items, 20 sequential draws, 16-frame image list) | Qwen3-VL-8B | 7 h 44 min | STATE.md item 1, task 39982137_0 |
| A3, 2,430 cells | Gemma 4 / Qwen3-VL | 0.82 / 1.12 GPU-h | `PREREG_A3_DRAFT.md`, Cost |
| Vision gates, bfloat16 stage | Gemma 4 / Qwen3-VL | 321.0 s / 6,257.9 s | `vision_gates_*.json`, `stages.bfloat16.seconds` |
| Readout ladder | Gemma 4 / Qwen3-VL | 655.6 s / 1,080.4 s | `readout_ladder_*.json`, `seconds` |
| A2 cell time, 243 items | Gemma 4 / Qwen3-VL | 11.6 / 16.5 GPU-h; peak 18.3 / 23.5 GB | `PREREG_A2_DRAFT.md`, Cost |
| A4 per model (upper bound) | Gemma 4 / Qwen3-VL | about 60 / 61 GPU-h | `PREREG_A4_DRAFT.md`, Cost |
| A1 rotation task | Gemma 4 E4B | not found in the repo | read from Slurm accounting before the freeze |

## Updates after the lead decisions and the audit (2026-11-09)

- Gemma 4 31B: the user approved storage on the lab project store; it is admitted inside the design block. Not yet downloaded.
- `ampere-long` (read-only, `scontrol show partition ampere-long`): `AllowQos=gpul`, one node (`gpu-q-1`), no time limit. `sacctmgr show assoc user=eb2007`: `baron-cohen-sl3-gpu` on `ampere` with QOS `gpu2,intr` only; `gpu2` MaxWall 12:00:00. So `ampere-long` is not available to this account.
- Cluster code (read-only `sha256sum` in `~/rds/hpc-work/study3`): every file the A1 job loads matches the local copy and the block; `scripts/run_stimulus_interventions.py` differs (`19b1fdae...`), pre-sync, as ROADMAP 2.1 expects.
- Software versions recorded by the siblings' gate runs (`research/reproduction/vision_gates_{qwen3vl,gemma4}.json`, `versions`): cv2 4.11.0, transformers 5.10.1, torch 2.10.0+cu128, python 3.10.4. numpy was not recorded. Delivery and ladder outputs record no versions.
- Rotation lookups: regenerated off-repo with the frozen `write_rotated_human_lookup` to obtain the pinned hashes (rot0 `07ab51da...`, rot1 `21b55baa...`, rot2 `ab6f6743...`, rot3 `84cad8f1...`, rot4 `c8c019b3...`). The cluster copies were not hashed.
