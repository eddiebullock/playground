#!/usr/bin/env python3
"""Export CHB manuscript figures from existing analysis outputs only.

Does not recompute inferential statistics. Plot values come from:
  - analysis_outputs/statistical_analysis.json (aggregate accuracies, CIs, tests)
  - results/full_run trial CSVs via load_results_for_analysis (per-state plots)

Figure numbering matches the manuscript Results section:
  Figure 1 — EU accuracy vs human benchmarks
  Figure 2 — EU per-mental-state accuracy heatmap
  Figure 3 — Mindreading performance tiers (359 states, fair subset)
  Figure S1–S4 — supplementary; S5 skipped (no upright/inverted data)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "full_run"
ANALYSIS = REPO / "analysis_outputs"
OUT_MAIN = REPO / "figures_final"
OUT_SUPP = REPO / "figures_supplementary"

sys.path.insert(0, str(REPO))
from analysis.load_results import load_results_for_analysis  # noqa: E402
from analysis.mr_fair_subset import fair_mr_video_trial_ids  # noqa: E402

MODELS = ["gemini-3-flash", "gpt-5", "gpt-5-mini", "claude-opus-4-5"]
DISPLAY = {
    "gemini-3-flash": "Gemini 3 Flash",
    "gpt-5": "GPT-5",
    "gpt-5-mini": "GPT-5 Mini",
    "claude-opus-4-5": "Claude Opus 4.5",
}
MODEL_ORDER = [DISPLAY[m] for m in MODELS]

mpl.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save_main(fig: plt.Figure, stem: str) -> None:
    OUT_MAIN.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_MAIN / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_MAIN / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


def save_supp(fig: plt.Figure, stem: str) -> None:
    OUT_SUPP.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_SUPP / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png")


def eu_video(df: pd.DataFrame) -> pd.DataFrame:
    d = df[
        (df["dataset"] == "eu_emotion")
        & (df["condition"] == "video_only")
        & (df["model"].isin(MODELS))
        & df["is_correct"].notna()
    ].copy()
    return d[d["correct_label"].astype(str).str.casefold() != "neutral"]


def mr_fair(df: pd.DataFrame, fair_ids: set[str]) -> pd.DataFrame:
    return df[
        (df["dataset"] == "mindreading")
        & (df["condition"] == "video_only")
        & (df["model"].isin(MODELS))
        & (df["trial_id"].astype(str).isin(fair_ids))
        & df["is_correct"].notna()
    ].copy()


def assign_tier(m: float) -> str:
    if m == 1.0:
        return "Perfect (100%)"
    if m == 0.0:
        return "Zero (0%)"
    if m >= 0.75:
        return "High (75–99%)"
    if m >= 0.50:
        return "Moderate (50–74%)"
    return "Low (1–49%)"


TIER_ORDER = [
    "Perfect (100%)",
    "High (75–99%)",
    "Moderate (50–74%)",
    "Low (1–49%)",
    "Zero (0%)",
]


def figure_1(stats: dict) -> None:
    """EU model accuracy vs human benchmarks (manuscript Figure 1)."""
    rows = []
    for m in MODELS:
        cell = stats["per_model_dataset"][m]["eu_emotion"]["video_only"]
        hb = stats["comparisons"]["vs_human_benchmark"][m]["eu_emotion"]["video_only"]
        rows.append(
            {
                "label": DISPLAY[m],
                "panel": "Video-only",
                "acc": cell["accuracy"] * 100,
                "lo": cell["wilson_ci_95"][0] * 100,
                "hi": cell["wilson_ci_95"][1] * 100,
                "p": hb["p_value_raw"],
                "is_human": False,
            }
        )
    rows.append(
        {
            "label": "Human\n(O'Reilly facial)",
            "panel": "Video-only",
            "acc": 63.0,
            "lo": np.nan,
            "hi": np.nan,
            "p": None,
            "is_human": True,
        }
    )
    flash_a = stats["per_model_dataset"]["gemini-3-flash"]["eu_emotion"]["audio_only"]
    hb_a = stats["comparisons"]["vs_human_benchmark"]["gemini-3-flash"]["eu_emotion"][
        "audio_only"
    ]
    rows.append(
        {
            "label": "Gemini 3 Flash\n(audio-only)",
            "panel": "Audio-only",
            "acc": flash_a["accuracy"] * 100,
            "lo": flash_a["wilson_ci_95"][0] * 100,
            "hi": flash_a["wilson_ci_95"][1] * 100,
            "p": hb_a["p_value_raw"],
            "is_human": False,
        }
    )
    rows.append(
        {
            "label": "Human\n(Lassalle vocal)",
            "panel": "Audio-only",
            "acc": 45.19,
            "lo": np.nan,
            "hi": np.nan,
            "p": None,
            "is_human": True,
        }
    )
    plot_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True, gridspec_kw={"width_ratios": [5, 2]})
    palette = sns.color_palette("viridis", n_colors=6)

    for ax, panel, colors in zip(
        axes,
        ["Video-only", "Audio-only"],
        [palette[:5], palette[4:]],
    ):
        sub = plot_df[plot_df["panel"] == panel].reset_index(drop=True)
        x = np.arange(len(sub))
        bar_colors = []
        for i, r in sub.iterrows():
            bar_colors.append("#8c8c8c" if r["is_human"] else colors[i % len(colors)])
        ax.bar(x, sub["acc"], color=bar_colors, edgecolor="black", linewidth=0.4, width=0.72)
        for i, r in sub.iterrows():
            if np.isfinite(r["lo"]):
                ax.errorbar(
                    i,
                    r["acc"],
                    yerr=[[r["acc"] - r["lo"]], [r["hi"] - r["acc"]]],
                    fmt="none",
                    ecolor="black",
                    capsize=2.5,
                    lw=0.8,
                )
            if r["p"] is not None:
                if r["p"] < 0.05:
                    mark = "*"
                elif r["p"] < 0.06:
                    mark = "\u2020"  # dagger for marginal (Claude p=.051)
                else:
                    mark = ""
                if mark:
                    y = (r["hi"] if np.isfinite(r["hi"]) else r["acc"]) + 2.0
                    ax.text(i, y, mark, ha="center", va="bottom", fontsize=9)
        ax.axhline(25, color="0.45", ls="--", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["label"], rotation=0, ha="center", fontsize=6.5)
        ax.set_title(panel, fontsize=9)
        ax.set_ylim(0, 100)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Accuracy (%)")
    axes[0].text(0.02, 0.97, "Chance = 25%", transform=axes[0].transAxes, va="top", fontsize=6.5, color="0.35")
    fig.suptitle("EU-Emotion accuracy vs human benchmarks", fontsize=10, y=1.02)
    save_main(fig, "Figure_1")


def figure_2(df: pd.DataFrame) -> dict:
    """EU per-mental-state accuracy heatmap (manuscript Figure 2)."""
    eu = eu_video(df)
    pivot = (
        eu.groupby(["correct_label", "model"])["is_correct"]
        .mean()
        .unstack()
        .reindex(columns=MODELS)
        .astype(float)
    )
    pivot.columns = [DISPLAY[c] for c in pivot.columns]
    pivot["M"] = pivot[MODEL_ORDER].mean(axis=1)
    pivot["SD"] = pivot[MODEL_ORDER].std(axis=1, ddof=0)
    pivot = pivot.sort_values("M", ascending=False)
    heat = pivot[MODEL_ORDER] * 100.0

    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    sns.heatmap(
        heat,
        ax=ax,
        cmap="cividis",
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".0f",
        annot_kws={"size": 7},
        cbar_kws={"label": "Accuracy (%)", "shrink": 0.8},
        linewidths=0.25,
        linecolor="white",
    )
    # Annotate M / SD to the right of the heatmap
    for i, (state, row) in enumerate(pivot.iterrows()):
        ax.text(
            len(MODEL_ORDER) + 0.15,
            i + 0.5,
            f"M={row['M']*100:.0f}  SD={row['SD']*100:.0f}",
            va="center",
            ha="left",
            fontsize=6.5,
            clip_on=False,
        )
    ax.set_xlabel("")
    ax.set_ylabel("Mental state")
    ax.set_title("EU-Emotion video-only accuracy by mental state")
    plt.subplots_adjust(right=0.78)
    save_main(fig, "Figure_2")
    return {
        "n_states": int(len(pivot)),
        "hardest": pivot["M"].nsmallest(4).index.tolist(),
        "easiest": pivot["M"].nlargest(4).index.tolist(),
    }


def figure_3(df: pd.DataFrame, fair_ids: set[str]) -> dict:
    """Mindreading performance tiers (manuscript Figure 3)."""
    mr = mr_fair(df, fair_ids)
    acc = mr.groupby(["correct_label", "model"])["is_correct"].mean().unstack().astype(float)
    mean = acc.mean(axis=1)
    tiers = mean.map(assign_tier)
    counts = tiers.value_counts().reindex(TIER_ORDER).fillna(0).astype(int)
    pct = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(3.5, 3.9))
    colors = sns.color_palette("cividis", n_colors=5)
    ax.bar(range(len(TIER_ORDER)), counts.values, color=colors, edgecolor="black", lw=0.4)
    ax.set_xticks(range(len(TIER_ORDER)))
    ax.set_xticklabels(TIER_ORDER, rotation=28, ha="right")
    ax.set_ylabel("Number of mental states")
    ymax = max(counts.values) * 1.28 if len(counts) else 1
    ax.set_ylim(0, ymax)
    for i, (c, p) in enumerate(zip(counts.values, pct.values)):
        ax.text(i, c + ymax * 0.02, f"{c}\n({p:.1f}%)", ha="center", va="bottom", fontsize=6.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Mindreading performance tiers")
    save_main(fig, "Figure_3")
    return {
        "n_states": int(len(mean)),
        "n_trials": int(mr["trial_id"].nunique()),
        "counts": counts.to_dict(),
    }


def figure_s1(df: pd.DataFrame) -> None:
    eu = eu_video(df)
    g = (
        eu.groupby(["correct_label", "model"])["is_correct"]
        .mean()
        .unstack()
        .reindex(columns=MODELS)
        .astype(float)
        * 100
    )
    g = g.loc[g.mean(axis=1).sort_values(ascending=True).index]
    g = g.rename(columns=DISPLAY)

    fig, ax = plt.subplots(figsize=(9, max(6.5, 0.32 * len(g))))
    g.plot(kind="barh", ax=ax, colormap="viridis", width=0.82, edgecolor="none")
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 100)
    ax.axvline(25, color="0.45", ls="--", lw=0.7, label="Chance (25%)")
    ax.legend(frameon=False, loc="lower right", title="")
    ax.set_ylabel("")
    ax.set_title("EU-Emotion video-only: per-mental-state accuracy (Neutral excluded)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_supp(fig, "Figure_S1")


def figure_s2(df: pd.DataFrame, fair_ids: set[str]) -> None:
    mr = mr_fair(df, fair_ids)
    g = (
        mr.groupby(["correct_label", "model"])["is_correct"]
        .mean()
        .unstack()
        .reindex(columns=MODELS)
        .astype(float)
        * 100
    )
    g["mean"] = g.mean(axis=1)
    g = g.sort_values("mean")
    heat = g[MODELS].rename(columns=DISPLAY)

    fig, ax = plt.subplots(figsize=(7.5, 48))
    sns.heatmap(
        heat,
        ax=ax,
        cmap="cividis",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Accuracy (%)", "shrink": 0.15},
        yticklabels=True,
        xticklabels=True,
    )
    ax.tick_params(axis="y", labelsize=4.5)
    ax.tick_params(axis="x", labelsize=8, rotation=30)
    ax.set_xlabel("")
    ax.set_ylabel("Mental state")
    ax.set_title(
        "Mindreading video-only fair subset: per-mental-state accuracy "
        f"(N={mr['trial_id'].nunique()} trials, {len(g)} states)"
    )
    save_supp(fig, "Figure_S2")


def figure_s3(stats: dict) -> None:
    conds = ["audio_only", "video_only", "multimodal"]
    cond_lab = {
        "audio_only": "Audio-only",
        "video_only": "Video-only",
        "multimodal": "Multimodal",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6), sharey=True)
    colors = sns.color_palette("viridis", 3)
    for ax, ds, title in zip(
        axes,
        ["eu_emotion", "mindreading"],
        ["EU-Emotion", "Mindreading"],
    ):
        accs, los, his = [], [], []
        for c in conds:
            cell = stats["per_model_dataset"]["gemini-3-flash"][ds][c]
            accs.append(cell["accuracy"] * 100)
            los.append(cell["wilson_ci_95"][0] * 100)
            his.append(cell["wilson_ci_95"][1] * 100)
        x = np.arange(3)
        ax.bar(x, accs, color=colors, edgecolor="black", lw=0.4)
        ax.errorbar(
            x,
            accs,
            yerr=[np.array(accs) - np.array(los), np.array(his) - np.array(accs)],
            fmt="none",
            ecolor="black",
            capsize=3,
            lw=0.8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([cond_lab[c] for c in conds], rotation=15, ha="right")
        ax.set_title(title)
        ax.axhline(25, color="0.45", ls="--", lw=0.7)
        ax.set_ylim(0, 100)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for i, a in enumerate(accs):
            ax.text(i, a + 3.5, f"{a:.1f}%", ha="center", fontsize=7)
    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle("Gemini 3 Flash modality ablations", fontsize=10, y=1.02)
    save_supp(fig, "Figure_S3")


def figure_s4(stats: dict) -> None:
    rows = stats["comparisons"]["cross_dataset_video_only"]
    # stable model order
    order = {m: i for i, m in enumerate(MODELS)}
    rows = sorted(rows, key=lambda r: order[r["model"]])
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(rows))
    w = 0.36
    eu = [r["eu_accuracy"] * 100 for r in rows]
    mr = [r["mindreading_accuracy"] * 100 for r in rows]
    labels = [DISPLAY[r["model"]] for r in rows]
    c1, c2 = sns.color_palette("viridis", 2)
    ax.bar(x - w / 2, eu, w, label="EU-Emotion", color=c1, edgecolor="black", lw=0.3)
    ax.bar(x + w / 2, mr, w, label="Mindreading (fair)", color=c2, edgecolor="black", lw=0.3)
    for i, r in enumerate(rows):
        p = r["p_value_bonferroni"]
        if p < 0.001:
            stars = "***"
        elif p < 0.01:
            stars = "**"
        elif p < 0.05:
            stars = "*"
        else:
            stars = ""
        if stars:
            ax.text(i, max(eu[i], mr[i]) + 2.5, stars, ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.axhline(25, color="0.45", ls="--", lw=0.7)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Cross-dataset generalisation (video-only)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_supp(fig, "Figure_S4")


def write_captions(fig2_meta: dict, fig3_meta: dict) -> None:
    OUT_MAIN.mkdir(parents=True, exist_ok=True)
    OUT_SUPP.mkdir(parents=True, exist_ok=True)
    text = f"""# Figure captions

Computers in Human Behavior manuscript figures. Asterisk convention: * p < .05, ** p < .01, *** p < .001 (Bonferroni-corrected where stated in Results). Chance performance = 25% (4-AFC). Accuracies use Wilson 95% confidence intervals unless noted.

## Main text

**Figure 1.** Model accuracy on EU-Emotion compared to human benchmarks.

*Note.* Left: video-only accuracies for Gemini 3 Flash, GPT-5, GPT-5 Mini, and Claude Opus 4.5 (*N* = 118 trials per model) versus the O’Reilly et al. (2016) facial-expression validation benchmark (63.00%, *N* = 1,231). Right: Gemini 3 Flash audio-only accuracy (*N* = 118) versus the Lassalle et al. (2019) UK vocal-expression benchmark (45.19%, *N* = 427). Error bars are Wilson 95% CIs for model accuracies. * *p* < .05 versus the modality-matched human benchmark; † marginal (*p* = .051). Dashed line = chance (25%). Human bars are point estimates without model CIs.

**Figure 2.** Model accuracy by mental state on the EU-Emotion task (video-only).

*Note.* Cell colour encodes per-model accuracy (%) for each mental state (cividis scale; 0% = light, 100% = dark). Rows sorted by mean accuracy across the four models (descending). M = mean accuracy across models; SD = standard deviation across models. Neutral excluded (*n* = {fig2_meta['n_states']} states). Trial counts per state ranged from 2–8 per model. Most difficult states included Unfriendly, Jealous, Interested, and Angry; easiest included Sad, Sad Low Intensity, Disgusted, and Worried (all *M* = 100%).

**Figure 3.** Performance tiers across 359 mental states on the Mindreading task (video-only).

*Note.* Performance tiers classify {fig3_meta['n_states']} mental states (*N* = {fig3_meta['n_trials']} trials per model; fair video-evaluated subset) by mean accuracy across Gemini 3 Flash, GPT-5, GPT-5 Mini, and Claude Opus 4.5. Tier counts: Perfect = {fig3_meta['counts'].get('Perfect (100%)', 0)}; High = {fig3_meta['counts'].get('High (75–99%)', 0)}; Moderate = {fig3_meta['counts'].get('Moderate (50–74%)', 0)}; Low = {fig3_meta['counts'].get('Low (1–49%)', 0)}; Zero = {fig3_meta['counts'].get('Zero (0%)', 0)}. T-marker audio-only trials without video input are excluded.

## Supplementary

**Figure S1.** Full per-mental-state accuracy on EU-Emotion (video-only), all four models.

*Note.* Horizontal bars show accuracy (%) for each non-neutral mental state and intensity variant under video-only input (*N* = 118 trials per model). Models: Gemini 3 Flash, GPT-5, GPT-5 Mini, Claude Opus 4.5. Dashed line = chance (25%). Neutral excluded. Source: `results/full_run/*_eu_emotion_video_only_results.csv` (not the pooled `per_emotion_breakdown.csv`, which mixes conditions and lacks a dataset column).

**Figure S2.** Full per-mental-state accuracy on Mindreading (video-only fair subset), all four models.

*Note.* Heatmap of accuracy (%) for each of {fig3_meta['n_states']} mental states × four models on the fair video-evaluated subset (*N* = {fig3_meta['n_trials']} trials per model). Rows sorted by mean accuracy ascending. Colour scale: cividis (0–100%). This is the full-resolution companion to main-text Figure 3.

**Figure S3.** Modality ablation comparison for Gemini 3 Flash across EU-Emotion and Mindreading.

*Note.* Accuracy (%) with Wilson 95% CIs under audio-only, video-only, and multimodal input. EU-Emotion: *N* = 118 per condition. Mindreading: audio-only *N* = 1,263; multimodal *N* = 1,240 valid; video-only fair subset *N* = 581. Pairwise modality contrasts and Bonferroni-corrected *p*-values are reported in Results / Table 3. Mindreading audio-inclusive conditions should be interpreted with the spoken-label confound caveat.

**Figure S4.** Cross-dataset generalisation (EU-Emotion vs Mindreading), video-only, all four models.

*Note.* Paired bars show video-only accuracy on EU-Emotion (*N* = 118) versus Mindreading fair subset (*N* = 581). Asterisks mark Bonferroni-corrected two-proportion *z*-tests (* *p* < .05). Only Claude Opus 4.5 showed a significant EU–Mindreading difference after correction (−14.55 pp, *p*_bonf = .013).

**Figure S5.** *Not generated.* No upright/inverted orientation condition is present in `results/full_run/*_results.csv`. Available conditions are video_only, audio_only, and multimodal only.
"""
    path = OUT_MAIN / "captions.md"
    path.write_text(text, encoding="utf-8")
    # also copy to supplementary folder for convenience
    (OUT_SUPP / "captions.md").write_text(text, encoding="utf-8")
    print(f"  wrote {path}")


def main() -> None:
    print("Loading trial results (same pipeline as Results)...")
    _, df, meta = load_results_for_analysis(RESULTS)
    fair_ids = fair_mr_video_trial_ids(df, models=MODELS)
    stats = json.loads((ANALYSIS / "statistical_analysis.json").read_text())

    # Sanity checks against Table 2 / 3
    assert stats["per_model_dataset"]["gemini-3-flash"]["eu_emotion"]["video_only"]["n_correct"] == 88
    assert stats["per_model_dataset"]["gpt-5"]["mindreading"]["video_only"]["n_total"] == 581
    assert len(fair_ids) == 581
    print(f"  fair subset n={len(fair_ids)}; {meta['mr_video_only_fair_subset']['note'][:80]}...")

    print("\nMain figures:")
    figure_1(stats)
    fig2_meta = figure_2(df)
    fig3_meta = figure_3(df, fair_ids)

    print("\nSupplementary figures:")
    figure_s1(df)
    figure_s2(df, fair_ids)
    figure_s3(stats)
    figure_s4(stats)
    print("  skipped Figure_S5 (no upright/inverted data)")

    print("\nCaptions:")
    write_captions(fig2_meta, fig3_meta)
    print("\nDone.")
    print(f"  main: {OUT_MAIN}")
    print(f"  supp: {OUT_SUPP}")


if __name__ == "__main__":
    main()
