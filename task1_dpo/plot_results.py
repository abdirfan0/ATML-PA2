from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/task1_dpo")
    parser.add_argument("--figure-dir", default="report/figures")
    args = parser.parse_args()

    results = Path(args.results_dir)
    figures = Path(args.figure_dir)
    figures.mkdir(parents=True, exist_ok=True)

    def read(filename):
        return json.loads((results / filename).read_text())

    def save(fig, name):
        fig.savefig(figures / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(figures / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Full quantitative table, including the different training budgets.
    rows = []
    for name in ["standard", "beta_0p03", "beta_0p10", "beta_0p30"]:
        evaluation = read(f"{name}_eval_summary.json")
        training = read(f"{name}_summary.json")
        rows.append({
            **evaluation,
            "training_pairs": training["num_examples"],
            "optimizer_updates": training["optimizer_updates"],
            "skipped_updates": training["skipped_updates"],
            "mean_training_loss": training["mean_training_loss"],
        })

    table = pd.DataFrame(rows)
    table.to_csv(results / "dpo_comparison.csv", index=False)

    # Plot only the equal-budget beta forks.
    sweep = table[table["name"] != "standard"].sort_values("beta")
    x = np.arange(len(sweep))
    labels = [f"{v:.2f}" for v in sweep["beta"]]

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), layout="constrained")
    metrics = [
        ("heldout_preference_accuracy", "Preference accuracy (%)", 100),
        ("sampled_kl_token_mean", "Sampled KL estimate / token", 1),
        ("reward_model_score_mean", "Mean reward-model score", 1),
        ("response_length_mean", "Mean response length (tokens)", 1),
    ]

    for ax, (key, title, scale) in zip(axes.flat, metrics):
        ax.plot(x, sweep[key].to_numpy() * scale, marker="o")
        ax.set_xticks(x, labels)
        ax.set_xlabel("Beta")
        ax.set_title(title)
        ax.grid(alpha=0.2)
        if key == "response_length_mean":
            ax.set_ylim(0, 256)
        if key == "heldout_preference_accuracy":
            ax.set_ylim(0, 100)
        if key == "sampled_kl_token_mean":
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axes[1, 1].set_ylim(0, 256)
    fig.suptitle("DPO beta study: 600 training pairs per condition")
    save(fig, "task1_beta_comparison")

    length = read("length_study_summary.json")
    names = ["standard", "length_balanced"]
    display_names = ["Standard", "Length-balanced"]
    strata = ["preferred_longer", "length_matched", "rejected_longer"]
    stratum_labels = ["Preferred longer", "Length matched", "Rejected longer"]

    strata_rows = []
    word_rows = []
    for name in names:
        for stratum in strata:
            strata_rows.append({
                "policy": name,
                "stratum": stratum,
                **length[name]["strata"][stratum],
            })
        word_rows.append({
            "policy": name,
            **{k: v for k, v in length[name].items() if k != "strata"},
        })

    pd.DataFrame(strata_rows).to_csv(
        results / "length_strata_comparison.csv", index=False
    )
    pd.DataFrame(word_rows).to_csv(
        results / "word_limit_comparison.csv", index=False
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    x = np.arange(3)
    width = 0.36

    for j, name in enumerate(names):
        values = [
            100 * length[name]["strata"][s]["preference_accuracy"]
            for s in strata
        ]
        bars = axes[0].bar(
            x + (j - 0.5) * width, values, width,
            label=display_names[j],
        )
        axes[0].bar_label(bars, fmt="%.1f", fontsize=8)

    axes[0].set_xticks(x, stratum_labels, rotation=15)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Preference accuracy (%)")
    axes[0].set_title("Length-stratified held-out pairs")
    axes[0].legend(fontsize=8)

    compliance = [100 * length[n]["word_limit_compliance"] for n in names]
    bars = axes[1].bar(display_names, compliance, color=["C0", "C1"])
    axes[1].bar_label(bars, fmt="%.0f%%")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Word-limit compliance")
    axes[1].set_ylabel("Compliant responses (%)")

    means = [length[n]["word_prompt_response_length_mean"] for n in names]
    stds = [length[n]["word_prompt_response_length_std"] for n in names]
    axes[2].bar(
        display_names, means, yerr=stds, capsize=5, color=["C0", "C1"]
    )
    axes[2].set_ylabel("Generated response tokens")
    axes[2].set_title("Response length: mean ± SD")

    fig.suptitle(
        "Length-control comparison; generation metrics use 10 common prompts"
    )
    save(fig, "task1_length_comparison")

    print("Saved figures to:", figures)
    print("Saved three CSV tables to:", results)


if __name__ == "__main__":
    main()
