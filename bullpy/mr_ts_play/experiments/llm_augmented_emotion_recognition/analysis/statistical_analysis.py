import json
from pathlib import Path
from itertools import combinations
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint, proportions_ztest, binom_test
from statsmodels.stats.power import zt_ind_solve_power
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
import seaborn as sns


class GolanReplicationStatistics:
    """
    Statistical analysis replicating Golan et al. (2006) methodology
    for LLM emotion recognition evaluation.

    This class is designed to:
      - Consume existing summary/per-emotion/prediction JSONs produced by the
        experiment scripts (EU-Emotion and MindReading).
      - Produce JSON/CSV/Markdown/figure outputs that mirror the structure,
        statistical tests, and narrative style of Golan et al. (2006),
        adapted for LLMs (binomial variance, single deterministic run).
    """

    def __init__(self, results_dir: str = "results", output_dir: str = "results/statistical_analysis") -> None:
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Human benchmarks from Golan et al. (2006) - CAM Battery (overall scores)
        # Values are proportions (0–1).
        self.human_benchmarks: Dict[str, Dict[str, float]] = {
            "control": {
                "n": 17,
                "overall_accuracy": 0.8629,
                "overall_sd": 0.0599,
                "facial_accuracy": 0.8706,
                "facial_sd": 0.0806,
                "vocal_accuracy": 0.8552,
                "vocal_sd": 0.0478,
            },
            "as_group": {
                "n": 21,
                "overall_accuracy": 0.6805,
                "overall_sd": 0.1169,
                "facial_accuracy": 0.6466,
                "facial_sd": 0.1592,
                "vocal_accuracy": 0.7142,
                "vocal_sd": 0.0924,
            },
        }

        # Chance performance for 4-AFC
        self.chance_level: float = 0.25

    # -------------------------------------------------------------------------
    # Loading and organization of results
    # -------------------------------------------------------------------------

    def load_all_results(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Load all EU-Emotion and MindReading results from the results directory.

        Expected layout:
          results/
            eu_emotion_<model_label>/
              summary.json
              per_emotion.json (optional but preferred)
              predictions.json (optional, required for McNemar)
            mindreading_<model_label>/
              summary.json
              per_emotion.json
              predictions.json

        Returns
        -------
        Dict with structure:
          {
            'eu_emotion': {
               model_label: {
                   'summary': {...},
                   'per_emotion': {...} or None,
                   'predictions': [...] or None,
                   'dataset': 'eu_emotion'
               },
               ...
            },
            'mindreading': {
               model_label: {...},
               ...
            }
          }
        """
        results: Dict[str, Dict[str, Dict[str, Any]]] = {
            "eu_emotion": {},
            "mindreading": {},
        }

        if not self.results_dir.exists():
            return results

        for subdir in self.results_dir.iterdir():
            if not subdir.is_dir():
                continue

            name = subdir.name
            if name.startswith("eu_emotion_"):
                dataset_key = "eu_emotion"
                model_label = name.replace("eu_emotion_", "")
            elif name.startswith("mindreading_"):
                dataset_key = "mindreading"
                model_label = name.replace("mindreading_", "")
            else:
                continue

            summary_path = subdir / "summary.json"
            per_emotion_path = subdir / "per_emotion.json"
            predictions_path = subdir / "predictions.json"

            if not summary_path.exists():
                continue

            with summary_path.open("r") as f:
                summary = json.load(f)

            per_emotion = None
            if per_emotion_path.exists():
                with per_emotion_path.open("r") as f:
                    per_emotion = json.load(f)

            predictions = None
            if predictions_path.exists():
                with predictions_path.open("r") as f:
                    predictions = json.load(f)

            results[dataset_key][model_label] = {
                "summary": summary,
                "per_emotion": per_emotion,
                "predictions": predictions,
                "dataset": dataset_key,
                "path": str(subdir),
            }

        return results

    # -------------------------------------------------------------------------
    # Core helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def cohens_h(p1: float, p2: float) -> float:
        """
        Effect size for difference between two proportions
        h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
        """
        # Guard against edge cases at 0 or 1
        p1 = np.clip(p1, 1e-9, 1 - 1e-9)
        p2 = np.clip(p2, 1e-9, 1 - 1e-9)
        return 2.0 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

    @staticmethod
    def _interpret_comparison(model_acc: float, human_acc: float, p_value: float) -> str:
        """
        Generate interpretation text matching Golan et al. style.
        """
        if p_value >= 0.05:
            return "not significantly different from"
        if model_acc > human_acc:
            return "significantly higher than"
        return "significantly lower than"

    def test_above_chance(self, n_correct: int, n_total: int) -> float:
        """
        Binomial test for performance above chance (25% for 4-AFC).

        Replicates: "All participants scored above chance (p < .01, Binomial test)".
        """
        if n_total == 0:
            return 1.0
        return float(
            # statsmodels uses "larger"/"smaller"/"two-sided" rather than "greater"/"less"
            binom_test(n_correct, n_total, self.chance_level, alternative="larger")
        )

    # -------------------------------------------------------------------------
    # 1. Basic performance metrics
    # -------------------------------------------------------------------------

    def compute_basic_metrics(
        self, all_results: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> pd.DataFrame:
        """
        Compute overall accuracy, SD (binomial), 95% CI (Wilson),
        and binomial tests vs chance for each model/dataset.
        """
        records: List[Dict[str, Any]] = []

        for dataset, models in all_results.items():
            for model_label, payload in models.items():
                summary = payload["summary"]
                n_total = int(summary.get("valid_predictions", summary.get("processed", 0)))
                n_correct = int(summary.get("correct", 0))
                acc = float(summary.get("accuracy", n_correct / n_total if n_total else 0.0))

                if n_total > 0:
                    # Binomial SD and CI
                    sd = np.sqrt(acc * (1.0 - acc) / n_total)
                    ci_low, ci_high = proportion_confint(
                        count=n_correct,
                        nobs=n_total,
                        alpha=0.05,
                        method="wilson",
                    )
                    p_vs_chance = self.test_above_chance(n_correct, n_total)
                else:
                    sd = np.nan
                    ci_low, ci_high = (np.nan, np.nan)
                    p_vs_chance = np.nan

                records.append(
                    {
                        "model": model_label,
                        "dataset": dataset,
                        "n_trials": n_total,
                        "n_correct": n_correct,
                        "accuracy": acc,
                        "sd_binomial": sd,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_above_chance": p_vs_chance,
                    }
                )

        df = pd.DataFrame.from_records(records)
        return df

    # -------------------------------------------------------------------------
    # 2. Comparison to human benchmarks
    # -------------------------------------------------------------------------

    def compare_to_human_baseline(
        self, model_acc: float, model_n: int, human_group: str = "control"
    ) -> Dict[str, Any]:
        """
        Two-proportion z-test comparing LLM to human benchmark.

        We treat the human group as having n_human = 100 trials (as in Golan,
        mean over 100-item CAM battery), with mean accuracy given in the paper.
        """
        human = self.human_benchmarks[human_group]
        human_acc = human["overall_accuracy"]
        human_n = 100  # CAM battery: 100 items

        model_correct = int(round(model_acc * model_n))
        human_correct = int(round(human_acc * human_n))

        if model_n == 0:
            z_stat, p_value = np.nan, np.nan
        else:
            z_stat, p_value = proportions_ztest(
                [model_correct, human_correct],
                [model_n, human_n],
                alternative="two-sided",
            )

        h = self.cohens_h(model_acc, human_acc)
        interpretation = self._interpret_comparison(model_acc, human_acc, p_value)

        return {
            "z": float(z_stat),
            "p": float(p_value),
            "cohens_h": float(h),
            "interpretation": interpretation,
        }

    def build_golan_comparison_table(
        self, basic_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build Golan-style table comparing each model to control and AS groups.

        Columns:
          model, dataset, n_trials, accuracy, sd_binomial,
          z_vs_control, p_vs_control, h_vs_control, interp_vs_control,
          z_vs_as, p_vs_as, h_vs_as, interp_vs_as
        """
        rows: List[Dict[str, Any]] = []

        for _, row in basic_df.iterrows():
            model = row["model"]
            dataset = row["dataset"]
            n_trials = int(row["n_trials"])
            acc = float(row["accuracy"])
            sd = float(row["sd_binomial"])

            ctrl_res = self.compare_to_human_baseline(acc, n_trials, human_group="control")
            as_res = self.compare_to_human_baseline(acc, n_trials, human_group="as_group")

            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "n_trials": n_trials,
                    "accuracy": acc,
                    "sd": sd,
                    "z_vs_control": ctrl_res["z"],
                    "p_vs_control": ctrl_res["p"],
                    "h_vs_control": ctrl_res["cohens_h"],
                    "interp_vs_control": ctrl_res["interpretation"],
                    "z_vs_as": as_res["z"],
                    "p_vs_as": as_res["p"],
                    "h_vs_as": as_res["cohens_h"],
                    "interp_vs_as": as_res["interpretation"],
                }
            )

        df = pd.DataFrame(rows)
        return df

    # -------------------------------------------------------------------------
    # 3. Pairwise model comparisons (EU-Emotion; Fisher + Bonferroni)
    # -------------------------------------------------------------------------

    def pairwise_model_comparisons(
        self, eu_basic_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, float]:
        """
        Pairwise Fisher's exact tests with Bonferroni correction for EU-Emotion.
        """
        models_data: Dict[str, Dict[str, int]] = {}
        for _, row in eu_basic_df.iterrows():
            model = row["model"]
            n_total = int(row["n_trials"])
            n_correct = int(row["n_correct"])
            models_data[model] = {
                "n_total": n_total,
                "n_correct": n_correct,
            }

        model_names = list(models_data.keys())
        if len(model_names) < 2:
            return pd.DataFrame(), 0.05

        n_comparisons = len(list(combinations(model_names, 2)))
        bonferroni_alpha = 0.05 / n_comparisons

        results: List[Dict[str, Any]] = []
        for m1, m2 in combinations(model_names, 2):
            m1_correct = models_data[m1]["n_correct"]
            m1_incorrect = models_data[m1]["n_total"] - m1_correct
            m2_correct = models_data[m2]["n_correct"]
            m2_incorrect = models_data[m2]["n_total"] - m2_correct

            table = [[m1_correct, m1_incorrect], [m2_correct, m2_incorrect]]
            odds_ratio, p_value = stats.fisher_exact(table)

            p_bonf = min(1.0, p_value * n_comparisons)
            results.append(
                {
                    "model1": m1,
                    "model2": m2,
                    "odds_ratio": odds_ratio,
                    "p_uncorrected": p_value,
                    "p_bonferroni": p_bonf,
                    "bonferroni_alpha": bonferroni_alpha,
                    "significant_bonferroni": p_bonf < bonferroni_alpha,
                    "significant_uncorrected": p_value < 0.05,
                }
            )

        df = pd.DataFrame(results)
        return df, bonferroni_alpha

    # -------------------------------------------------------------------------
    # 4. Post-hoc power analysis
    # -------------------------------------------------------------------------

    def post_hoc_power_analysis(
        self, acc: float, n: int, human_group: str = "control", alpha: float = 0.01
    ) -> float:
        """
        Post-hoc power analysis matching Golan et al. (2006):
        "Power calculations for the different scales (with α = 0.01)".
        """
        human_acc = self.human_benchmarks[human_group]["overall_accuracy"]
        effect_size = abs(self.cohens_h(acc, human_acc))

        if n <= 0 or effect_size <= 0:
            return float("nan")

        # Two-sided z-test for independent proportions; ratio=1 for simplicity.
        power = zt_ind_solve_power(
            effect_size=effect_size,
            nobs1=n,
            alpha=alpha,
            ratio=1.0,
            alternative="two-sided",
        )
        return float(power)

    # -------------------------------------------------------------------------
    # 5. Per-emotion analysis
    # -------------------------------------------------------------------------

    def per_emotion_dataframe(
        self, all_results: Dict[str, Dict[str, Dict[str, Any]]], dataset: str
    ) -> pd.DataFrame:
        """
        Build a wide per-emotion accuracy table for a given dataset.

        Columns:
          emotion, <model1>_n, <model1>_correct, <model1>_acc, ...

        Also includes:
          p_vs_chance_<model>, sig_vs_chance_<model>, ceiling_<model>, floor_<model>
        """
        models = all_results.get(dataset, {})
        if not models:
            return pd.DataFrame()

        # Collect all emotions
        all_emotions: set = set()
        for payload in models.values():
            per_emotion = payload.get("per_emotion") or payload["summary"].get(
                "per_emotion"
            )
            if not per_emotion:
                continue
            all_emotions.update(per_emotion.keys())

        rows: List[Dict[str, Any]] = []
        for emotion in sorted(all_emotions):
            record: Dict[str, Any] = {"emotion": emotion}
            for model_label, payload in models.items():
                per_emotion = payload.get("per_emotion") or payload["summary"].get(
                    "per_emotion"
                )
                if not per_emotion or emotion not in per_emotion:
                    continue

                data = per_emotion[emotion]
                n = int(data.get("count", data.get("total", 0)))
                correct = int(data.get("correct", 0))
                acc = float(data.get("accuracy", correct / n if n else 0.0))

                record[f"{model_label}_n"] = n
                record[f"{model_label}_correct"] = correct
                record[f"{model_label}_acc"] = acc

                if n > 0:
                    p_vs_chance = self.test_above_chance(correct, n)
                else:
                    p_vs_chance = np.nan

                sig = ""
                if not np.isnan(p_vs_chance):
                    if p_vs_chance < 0.01:
                        sig = "**"
                    elif p_vs_chance < 0.05:
                        sig = "*"

                record[f"p_vs_chance_{model_label}"] = p_vs_chance
                record[f"sig_vs_chance_{model_label}"] = sig
                record[f"ceiling_{model_label}"] = acc >= 0.95 if n > 0 else False
                record[f"floor_{model_label}"] = acc <= 0.30 if n > 0 else False

            rows.append(record)

        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # 6. Modality effects (McNemar, MindReading multimodal vs video-only)
    # -------------------------------------------------------------------------

    def _find_mindreading_modality_pairs(
        self, mindreading_results: Dict[str, Dict[str, Any]]
    ) -> List[Tuple[str, str]]:
        """
        Heuristically group MindReading result directories into (multimodal, video-only)
        pairs for the same underlying model.

        Strategy:
          - Use the model label suffix before obvious markers like '_video', '_audio',
            or '_multimodal' when present.
          - Fall back to exact label match of prefixes where one has video_only==True
            and the other use_audio==True.
        """
        # First, group by a naive base key: split on common modality tokens.
        modality_tokens = ["_video_only", "_multimodal", "_audio", "_va", "_v"]

        def base_key(label: str) -> str:
            for tok in modality_tokens:
                if tok in label:
                    return label.replace(tok, "")
            return label

        by_base: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        for label, payload in mindreading_results.items():
            key = base_key(label)
            by_base.setdefault(key, []).append((label, payload))

        pairs: List[Tuple[str, str]] = []
        for key, entries in by_base.items():
            if len(entries) < 2:
                continue
            # Look for one with video_only True and one with use_audio True.
            video_labels = [
                lbl for lbl, p in entries if p["summary"].get("video_only") is True
            ]
            audio_labels = [
                lbl for lbl, p in entries if p["summary"].get("use_audio") is True
            ]
            for v_lbl in video_labels:
                for a_lbl in audio_labels:
                    pairs.append((a_lbl, v_lbl))

        return pairs

    def modality_mcnemar_tests(
        self, mindreading_results: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Run McNemar's test comparing multimodal vs video-only conditions
        for paired trials (same trial IDs).

        Returns empty DataFrame if predictions.json files or pairs are not available.
        """
        pairs = self._find_mindreading_modality_pairs(mindreading_results)
        if not pairs:
            return pd.DataFrame()

        records: List[Dict[str, Any]] = []
        for multimodal_label, video_label in pairs:
            mm_payload = mindreading_results[multimodal_label]
            vo_payload = mindreading_results[video_label]
            mm_preds = mm_payload.get("predictions")
            vo_preds = vo_payload.get("predictions")
            if not mm_preds or not vo_preds:
                continue

            # Map by trial_id
            mm_by_id = {p.get("trial_id"): p for p in mm_preds if p.get("trial_id") is not None}
            vo_by_id = {p.get("trial_id"): p for p in vo_preds if p.get("trial_id") is not None}
            common_ids = sorted(set(mm_by_id.keys()) & set(vo_by_id.keys()))
            if not common_ids:
                continue

            b = 0  # multimodal correct, video-only incorrect
            c = 0  # multimodal incorrect, video-only correct
            for tid in common_ids:
                mm_ok = mm_by_id[tid].get("is_correct")
                vo_ok = vo_by_id[tid].get("is_correct")
                if mm_ok is None or vo_ok is None:
                    continue
                if mm_ok and not vo_ok:
                    b += 1
                elif vo_ok and not mm_ok:
                    c += 1

            table = [[0, b], [c, 0]]
            # Use exact McNemar when possible
            result = mcnemar(table, exact=True) if (b + c) < 25 else mcnemar(table, exact=False)
            records.append(
                {
                    "model_multimodal": multimodal_label,
                    "model_video_only": video_label,
                    "b_multimodal_only_correct": b,
                    "c_video_only_only_correct": c,
                    "statistic": float(result.statistic),
                    "p_value": float(result.pvalue),
                }
            )

        return pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # 7. Cross-dataset generalization (EU vs MindReading)
    # -------------------------------------------------------------------------

    def cross_dataset_generalization(
        self, basic_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        For models that appear in both EU-Emotion and MindReading,
        compare accuracy using two-proportion z-tests and Cohen's h.
        """
        eu = basic_df[basic_df["dataset"] == "eu_emotion"]
        mr = basic_df[basic_df["dataset"] == "mindreading"]

        eu_map = {row["model"]: row for _, row in eu.iterrows()}
        mr_map = {row["model"]: row for _, row in mr.iterrows()}
        common_models = sorted(set(eu_map.keys()) & set(mr_map.keys()))

        records: List[Dict[str, Any]] = []
        for model in common_models:
            eu_row = eu_map[model]
            mr_row = mr_map[model]

            n_eu = int(eu_row["n_trials"])
            c_eu = int(eu_row["n_correct"])
            n_mr = int(mr_row["n_trials"])
            c_mr = int(mr_row["n_correct"])

            if n_eu == 0 or n_mr == 0:
                continue

            z_stat, p_val = proportions_ztest(
                [c_eu, c_mr],
                [n_eu, n_mr],
                alternative="two-sided",
            )

            acc_eu = float(eu_row["accuracy"])
            acc_mr = float(mr_row["accuracy"])
            h = self.cohens_h(acc_eu, acc_mr)

            records.append(
                {
                    "model": model,
                    "eu_n": n_eu,
                    "eu_acc": acc_eu,
                    "mr_n": n_mr,
                    "mr_acc": acc_mr,
                    "z": float(z_stat),
                    "p": float(p_val),
                    "cohens_h": float(h),
                }
            )

        return pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # 8. Variance and reliability analysis across emotions/models
    # -------------------------------------------------------------------------

    def variance_and_reliability(
        self, per_emotion_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute per-emotion variance across models and identify
        high-agreement and low-agreement emotions.

        Also compute a correlation matrix between model accuracy vectors.
        """
        if per_emotion_df.empty:
            return {
                "high_agreement_emotions": [],
                "low_agreement_emotions": [],
                "correlation_matrix": {},
            }

        # Identify model accuracy columns of the form "<model>_acc"
        acc_cols = [c for c in per_emotion_df.columns if c.endswith("_acc")]
        if not acc_cols:
            return {
                "high_agreement_emotions": [],
                "low_agreement_emotions": [],
                "correlation_matrix": {},
            }

        # Compute mean/variance across models per emotion
        acc_matrix = per_emotion_df[acc_cols].to_numpy(dtype=float)
        mean_acc = np.nanmean(acc_matrix, axis=1)
        var_acc = np.nanvar(acc_matrix, axis=1)

        per_emotion_df = per_emotion_df.copy()
        per_emotion_df["mean_acc"] = mean_acc
        per_emotion_df["var_acc"] = var_acc

        # High-agreement: all available model accuracies >= 0.80
        high_agreement: List[str] = []
        low_agreement: List[str] = []
        for idx, row in per_emotion_df.iterrows():
            values = [
                row[col] for col in acc_cols if not np.isnan(row[col])
            ]
            if not values:
                continue
            if all(v >= 0.80 for v in values):
                high_agreement.append(row["emotion"])
            if all(v <= 0.60 for v in values):
                low_agreement.append(row["emotion"])

        # Correlation matrix (models x models across emotions)
        model_names = [c[:-4] for c in acc_cols]  # strip "_acc"
        corr_matrix: Dict[str, Dict[str, float]] = {}
        if len(model_names) >= 2:
            acc_df = per_emotion_df[acc_cols]
            corr = acc_df.corr(method="pearson")
            corr.index = model_names
            corr.columns = model_names
            corr_matrix = corr.to_dict()

        return {
            "high_agreement_emotions": high_agreement,
            "low_agreement_emotions": low_agreement,
            "correlation_matrix": corr_matrix,
            "per_emotion_with_stats": per_emotion_df.to_dict(orient="records"),
        }

    # -------------------------------------------------------------------------
    # Reporting and figure generation
    # -------------------------------------------------------------------------

    @staticmethod
    def _format_p(p: float) -> str:
        if np.isnan(p):
            return "n.s."
        if p < 0.001:
            return "p < .001"
        return f"p = {p:.3f}".replace("0.", ".")

    @staticmethod
    def _sig_stars(p: float) -> str:
        if np.isnan(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def generate_golan_style_report(
        self,
        basic_df: pd.DataFrame,
        comparison_df: pd.DataFrame,
        pairwise_df: pd.DataFrame,
        bonferroni_alpha: float,
        power_df: pd.DataFrame,
        eu_per_emotion_df: pd.DataFrame,
        mr_per_emotion_df: pd.DataFrame,
        modality_df: pd.DataFrame,
        cross_dataset_df: pd.DataFrame,
        variance_info: Dict[str, Any],
    ) -> str:
        """
        Generate markdown report in Golan et al. (2006) style.
        """
        lines: List[str] = []
        lines.append("# Statistical Analysis: LLM Emotion Recognition Performance")
        lines.append("")
        lines.append("## Replication of Golan et al. (2006) Methodology")
        lines.append("")

        # Section: Overall performance
        lines.append("### Overall Performance")
        lines.append(
            "All LLMs were evaluated on 4-alternative forced-choice emotion "
            "recognition tasks. Chance performance was 25%."
        )
        if not basic_df.empty and basic_df["p_above_chance"].notna().all():
            if (basic_df["p_above_chance"] < 0.01).all():
                lines.append(
                    "Across datasets, all model accuracies were significantly above "
                    "chance (p < .01, binomial test)."
                )
        lines.append("")

        # Section: Comparison to human benchmarks
        lines.append("### Comparison to Human Benchmarks")
        lines.append(
            "Two-proportion z-tests were conducted comparing each model's overall "
            "accuracy to the control group (M = 86.29%, SD = 5.99%) and AS group "
            "(M = 68.05%, SD = 11.69%) reported by Golan et al. (2006). "
            "Effect sizes were quantified using Cohen's h."
        )
        lines.append("")

        for _, row in comparison_df.iterrows():
            model = row["model"]
            dataset = row["dataset"]
            acc = row["accuracy"] * 100.0
            zc = row["z_vs_control"]
            pc = row["p_vs_control"]
            ic = row["interp_vs_control"]
            za = row["z_vs_as"]
            pa = row["p_vs_as"]
            ia = row["interp_vs_as"]

            lines.append(
                f"For {model} on {dataset}, accuracy was {acc:.1f}%. Compared to "
                f"controls, performance was {ic} (z = {zc:.2f}, {self._format_p(pc)}). "
                f"Compared to the AS group, performance was {ia} "
                f"(z = {za:.2f}, {self._format_p(pa)})."
            )
        lines.append("")

        # Section: Pairwise model comparisons
        lines.append("### Pairwise Model Comparisons (EU-Emotion)")
        if pairwise_df.empty:
            lines.append(
                "Pairwise Fisher's exact tests could not be computed because fewer "
                "than two EU-Emotion model results were available."
            )
        else:
            lines.append(
                "Pairwise Fisher's exact tests were conducted between all pairs of "
                "LLMs on the EU-Emotion dataset. A Bonferroni-corrected alpha of "
                f"{bonferroni_alpha:.4f} was used to control the family-wise error rate."
            )
            n_sig = int(pairwise_df["significant_bonferroni"].sum())
            lines.append(
                f"{n_sig} pairwise differences remained significant after Bonferroni "
                "correction."
            )
        lines.append("")

        # Section: Power analysis
        lines.append("### Power Analysis")
        if power_df.empty:
            lines.append(
                "Post-hoc power could not be estimated because no valid basic "
                "statistics were available."
            )
        else:
            lines.append(
                "Post-hoc power analyses (α = .01, two-sided) indicated high power "
                "to detect differences between model performance and human benchmarks."
            )
            for _, row in power_df.iterrows():
                lines.append(
                    f"For {row['model']} on {row['dataset']}, power to detect a "
                    f"difference from the control group was {row['power_vs_control']:.2f}, "
                    f"and from the AS group was {row['power_vs_as']:.2f}."
                )
        lines.append("")

        # Section: Per-emotion performance
        lines.append("### Per-Emotion Performance")
        if eu_per_emotion_df.empty and mr_per_emotion_df.empty:
            lines.append(
                "Per-emotion analyses could not be performed because per-emotion "
                "statistics were not available."
            )
        else:
            lines.append(
                "Per-emotion analyses (Table: per_emotion_performance.csv) revealed "
                "substantial variability in difficulty across emotion concepts. "
                "Emotions with ceiling effects (≥ 95% correct) and floor effects "
                "(≤ 30% correct) were identified for each model."
            )
        lines.append("")

        # Section: Modality effects
        lines.append("### Modality Effects (MindReading)")
        if modality_df.empty:
            lines.append(
                "No paired multimodal vs video-only MindReading runs with matching "
                "predictions were detected, so McNemar's test could not be applied."
            )
        else:
            lines.append(
                "McNemar's tests were used to compare multimodal (video + audio) "
                "and video-only conditions on the MindReading dataset."
            )
            for _, row in modality_df.iterrows():
                lines.append(
                    f"For {row['model_multimodal']} vs {row['model_video_only']}, "
                    f"McNemar's test yielded χ²(1) = {row['statistic']:.2f}, "
                    f"{self._format_p(row['p_value'])}."
                )
        lines.append("")

        # Section: Cross-dataset generalization
        lines.append("### Cross-Dataset Generalization")
        if cross_dataset_df.empty:
            lines.append(
                "No models were evaluated on both EU-Emotion and MindReading, so "
                "cross-dataset generalization could not be assessed."
            )
        else:
            lines.append(
                "Two-proportion z-tests compared performance on EU-Emotion (27 "
                "emotions) vs MindReading (425 emotions) for models evaluated on "
                "both datasets."
            )
            for _, row in cross_dataset_df.iterrows():
                direction = "decrease" if row["mr_acc"] < row["eu_acc"] else "increase"
                lines.append(
                    f"For {row['model']}, accuracy changed from "
                    f"{row['eu_acc']*100:.1f}% (EU-Emotion) to "
                    f"{row['mr_acc']*100:.1f}% (MindReading), "
                    f"representing a {direction} in performance "
                    f"(z = {row['z']:.2f}, {self._format_p(row['p'])})."
                )
        lines.append("")

        # Section: Variance and reliability
        lines.append("### Variance and Reliability Across Emotion Concepts")
        high_agree = variance_info.get("high_agreement_emotions", [])
        low_agree = variance_info.get("low_agreement_emotions", [])
        if not high_agree and not low_agree:
            lines.append(
                "Variance and reliability analyses could not be computed because "
                "per-emotion accuracy matrices were not available."
            )
        else:
            lines.append(
                "Consistency analyses across models identified sets of emotions "
                "with uniformly high agreement (≥ 80% accurate for all models) "
                "and uniformly low agreement (≤ 60% accurate for all models)."
            )
            if high_agree:
                lines.append(
                    "High-agreement emotions included: " + ", ".join(sorted(high_agree)) + "."
                )
            if low_agree:
                lines.append(
                    "Low-agreement emotions included: " + ", ".join(sorted(low_agree)) + "."
                )
        lines.append("")

        # Limitations
        lines.append("### Methodological Considerations and Limitations")
        lines.append(
            "Golan et al. (2006) evaluated human participants on a 100-item CAM "
            "battery covering 20 emotion concepts with both facial and vocal "
            "stimuli, whereas the present study evaluates LLMs on distinct "
            "emotion sets (27 EU-Emotion labels and 425 MindReading labels) and "
            "modality configurations (vision-only, video + audio)."
        )
        lines.append(
            "LLMs are deterministic (given fixed prompts and seeds), so between-"
            "trial variability is captured by binomial sampling rather than "
            "between-subject variance. Consequently, we model LLM variability "
            "using binomial standard errors, while human variability reflects "
            "individual differences."
        )
        lines.append(
            "All cross-study comparisons should therefore be interpreted with "
            "caution: the inferential framework matches Golan et al. as closely "
            "as possible, but the emotion vocabularies, modalities, and variance "
            "structures differ."
        )

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------

    def create_golan_style_figures(
        self,
        basic_df: pd.DataFrame,
        comparison_df: pd.DataFrame,
        pairwise_df: pd.DataFrame,
        eu_per_emotion_df: pd.DataFrame,
    ) -> None:
        """
        Create figures matching Golan et al. style:
          - performance_comparison.png
          - pairwise_heatmap.png
          - emotion_difficulty_ranking.png
        """
        if basic_df.empty:
            return

        # Figure 1: performance_comparison.png
        fig_path = self.output_dir / "figures"
        fig_path.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 5))
        plot_df = basic_df.copy()
        plot_df["accuracy_pct"] = plot_df["accuracy"] * 100.0

        # Add human benchmarks as pseudo-rows for plotting
        for dataset in sorted(plot_df["dataset"].unique()):
            for label, hb in self.human_benchmarks.items():
                plot_df = pd.concat(
                    [
                        plot_df,
                        pd.DataFrame(
                            [
                                {
                                    "model": f"human_{label}",
                                    "dataset": dataset,
                                    "n_trials": 100,
                                    "n_correct": int(round(hb["overall_accuracy"] * 100)),
                                    "accuracy": hb["overall_accuracy"],
                                    "sd_binomial": hb["overall_sd"],
                                    "ci_low": np.nan,
                                    "ci_high": np.nan,
                                    "p_above_chance": np.nan,
                                    "accuracy_pct": hb["overall_accuracy"] * 100.0,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

        sns.barplot(
            data=plot_df,
            x="model",
            y="accuracy_pct",
            hue="dataset",
        )
        plt.ylabel("Accuracy (%)")
        plt.xlabel("Model / Group")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(fig_path / "performance_comparison.png", dpi=300)
        plt.close()

        # Figure 2: pairwise_heatmap.png
        if not pairwise_df.empty:
            models = sorted(
                set(pairwise_df["model1"]).union(set(pairwise_df["model2"]))
            )
            matrix = pd.DataFrame(
                np.nan, index=models, columns=models, dtype=float
            )
            for _, row in pairwise_df.iterrows():
                m1 = row["model1"]
                m2 = row["model2"]
                p = row["p_uncorrected"]
                matrix.loc[m1, m2] = p
                matrix.loc[m2, m1] = p
            np.fill_diagonal(matrix.values, 0.0)

            plt.figure(figsize=(6, 5))
            sns.heatmap(
                matrix,
                annot=True,
                fmt=".3f",
                cmap="viridis_r",
                cbar_kws={"label": "Uncorrected p-value"},
            )
            plt.title("Pairwise Model Comparisons (Fisher's Exact, EU-Emotion)")
            plt.tight_layout()
            plt.savefig(fig_path / "pairwise_heatmap.png", dpi=300)
            plt.close()

        # Figure 3: emotion_difficulty_ranking.png
        if not eu_per_emotion_df.empty:
            acc_cols = [c for c in eu_per_emotion_df.columns if c.endswith("_acc")]
            if acc_cols:
                eu_per_emotion_df = eu_per_emotion_df.copy()
                eu_per_emotion_df["mean_acc"] = eu_per_emotion_df[acc_cols].mean(
                    axis=1
                )
                ranked = eu_per_emotion_df.sort_values("mean_acc")
                plt.figure(figsize=(8, max(4, 0.2 * len(ranked))))
                plt.barh(
                    ranked["emotion"],
                    ranked["mean_acc"] * 100.0,
                )
                plt.xlabel("Mean Accuracy Across Models (%)")
                plt.ylabel("Emotion")
                plt.tight_layout()
                plt.savefig(fig_path / "emotion_difficulty_ranking.png", dpi=300)
                plt.close()

    # -------------------------------------------------------------------------
    # Orchestration
    # -------------------------------------------------------------------------

    def run_full_analysis(self) -> None:
        """
        Execute complete statistical analysis pipeline and save all artifacts:
          - statistical_summary.json
          - golan_comparison_table.csv
          - pairwise_comparisons.csv
          - per_emotion_performance.csv
          - power_analysis.txt
          - statistical_report.md
          - figures/*.png
        """
        print("Loading results...")
        all_results = self.load_all_results()

        print("Computing basic performance metrics...")
        basic_df = self.compute_basic_metrics(all_results)

        print("Comparing to human benchmarks...")
        comparison_df = self.build_golan_comparison_table(basic_df)

        print("Conducting pairwise model comparisons (EU-Emotion)...")
        eu_basic = basic_df[basic_df["dataset"] == "eu_emotion"].copy()
        pairwise_df, bonf_alpha = self.pairwise_model_comparisons(eu_basic)

        print("Performing power analysis...")
        power_records: List[Dict[str, Any]] = []
        for _, row in basic_df.iterrows():
            model = row["model"]
            dataset = row["dataset"]
            acc = float(row["accuracy"])
            n_trials = int(row["n_trials"])
            power_ctrl = self.post_hoc_power_analysis(
                acc, n_trials, human_group="control", alpha=0.01
            )
            power_as = self.post_hoc_power_analysis(
                acc, n_trials, human_group="as_group", alpha=0.01
            )
            power_records.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "n_trials": n_trials,
                    "accuracy": acc,
                    "power_vs_control": power_ctrl,
                    "power_vs_as": power_as,
                }
            )
        power_df = pd.DataFrame(power_records)

        print("Analyzing per-emotion performance...")
        eu_per_emotion_df = self.per_emotion_dataframe(all_results, "eu_emotion")
        mr_per_emotion_df = self.per_emotion_dataframe(all_results, "mindreading")

        print("Assessing modality effects (MindReading)...")
        modality_df = self.modality_mcnemar_tests(all_results.get("mindreading", {}))

        print("Analyzing cross-dataset generalization...")
        cross_dataset_df = self.cross_dataset_generalization(basic_df)

        print("Computing variance and reliability metrics...")
        variance_info = self.variance_and_reliability(eu_per_emotion_df)

        # ------------------------------------------------------------------
        # Save tables and JSON summaries
        # ------------------------------------------------------------------
        print("Saving summary artifacts...")
        # 1. statistical_summary.json
        summary_payload: Dict[str, Any] = {
            "basic_metrics": basic_df.to_dict(orient="records"),
            "comparison_to_humans": comparison_df.to_dict(orient="records"),
            "pairwise_comparisons": pairwise_df.to_dict(orient="records"),
            "bonferroni_alpha": bonf_alpha,
            "power_analysis": power_df.to_dict(orient="records"),
            "per_emotion_eu": eu_per_emotion_df.to_dict(orient="records"),
            "per_emotion_mindreading": mr_per_emotion_df.to_dict(orient="records"),
            "modality_effects": modality_df.to_dict(orient="records"),
            "cross_dataset_generalization": cross_dataset_df.to_dict(orient="records"),
            "variance_and_reliability": variance_info,
        }
        with (self.output_dir / "statistical_summary.json").open("w") as f:
            json.dump(summary_payload, f, indent=2)

        # 2. golan_comparison_table.csv
        comparison_df.to_csv(self.output_dir / "golan_comparison_table.csv", index=False)

        # 3. pairwise_comparisons.csv
        pairwise_df.to_csv(self.output_dir / "pairwise_comparisons.csv", index=False)

        # 4. per_emotion_performance.csv
        per_emotion_combined = []
        if not eu_per_emotion_df.empty:
            eu_tmp = eu_per_emotion_df.copy()
            eu_tmp["dataset"] = "eu_emotion"
            per_emotion_combined.append(eu_tmp)
        if not mr_per_emotion_df.empty:
            mr_tmp = mr_per_emotion_df.copy()
            mr_tmp["dataset"] = "mindreading"
            per_emotion_combined.append(mr_tmp)
        if per_emotion_combined:
            per_emotion_all = pd.concat(per_emotion_combined, ignore_index=True)
            per_emotion_all.to_csv(
                self.output_dir / "per_emotion_performance.csv", index=False
            )
        else:
            # Write an empty file with header
            pd.DataFrame(columns=["emotion", "dataset"]).to_csv(
                self.output_dir / "per_emotion_performance.csv", index=False
            )

        # 5. power_analysis.txt
        with (self.output_dir / "power_analysis.txt").open("w") as f:
            for _, row in power_df.iterrows():
                f.write(
                    f"{row['model']} ({row['dataset']}): "
                    f"n={row['n_trials']}, "
                    f"power vs control = {row['power_vs_control']:.3f}, "
                    f"power vs AS = {row['power_vs_as']:.3f}\n"
                )

        # 6. statistical_report.md
        report_md = self.generate_golan_style_report(
            basic_df=basic_df,
            comparison_df=comparison_df,
            pairwise_df=pairwise_df,
            bonferroni_alpha=bonf_alpha,
            power_df=power_df,
            eu_per_emotion_df=eu_per_emotion_df,
            mr_per_emotion_df=mr_per_emotion_df,
            modality_df=modality_df,
            cross_dataset_df=cross_dataset_df,
            variance_info=variance_info,
        )
        with (self.output_dir / "statistical_report.md").open("w") as f:
            f.write(report_md)

        # 7. figures
        print("Creating visualizations...")
        self.create_golan_style_figures(
            basic_df=basic_df,
            comparison_df=comparison_df,
            pairwise_df=pairwise_df,
            eu_per_emotion_df=eu_per_emotion_df,
        )

        print(f"Analysis complete. Results saved to {self.output_dir}")


if __name__ == "__main__":
    analyzer = GolanReplicationStatistics()
    analyzer.run_full_analysis()

