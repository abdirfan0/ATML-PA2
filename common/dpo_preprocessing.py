from collections import Counter
from pathlib import Path

from common.data import (
    read_jsonl,
    prompt_messages_from_preference,
)
from common.logging_utils import save_json


def load_dpo_rows(path, tokenizer, max_length, max_examples=None):
    original_rows = read_jsonl(path)
    eligible_rows = []
    eligible_indices = []
    prompt_lengths = []
    excluded_indices = []

    for index, row in enumerate(original_rows):
        ids = tokenizer.apply_chat_template(
            prompt_messages_from_preference(row),
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt_lengths.append(len(ids))

        if len(ids) >= max_length:
            excluded_indices.append(index)
        else:
            eligible_rows.append(row)
            eligible_indices.append(index)

    # Apply the experiment budget after filtering.
    selected_rows = eligible_rows
    selected_indices = eligible_indices

    if max_examples is not None:
        selected_rows = selected_rows[:int(max_examples)]
        selected_indices = selected_indices[:int(max_examples)]

    # Record categorical length/stratum fields when supplied.
    stratum_counts = {}
    keys = set().union(*(row.keys() for row in original_rows))

    for key in sorted(keys):
        if "strat" in key.lower() or "length" in key.lower():
            values = [row.get(key) for row in original_rows]
            if any(isinstance(value, str) for value in values):
                stratum_counts[key] = {
                    "original": dict(Counter(
                        str(row.get(key)) for row in original_rows
                    )),
                    "retained": dict(Counter(
                        str(row.get(key)) for row in eligible_rows
                    )),
                    "selected": dict(Counter(
                        str(row.get(key)) for row in selected_rows
                    )),
                }

    info = {
        "dataset": str(path),
        "max_sequence_length": int(max_length),
        "rule": (
            "Exclude pairs whose prompt has >= max_sequence_length tokens; "
            "otherwise preserve prompt and right-truncate response, reserving EOS."
        ),
        "original_count": len(original_rows),
        "eligible_count": len(eligible_rows),
        "excluded_count": len(excluded_indices),
        "selected_count": len(selected_rows),
        "excluded_indices": excluded_indices,
        "eligible_indices": eligible_indices,
        "selected_indices": selected_indices,
        "prompt_token_lengths": prompt_lengths,
        "stratum_counts": stratum_counts,
    }

    manifest = (
        Path("results/task1_dpo/preprocessing")
        / f"{Path(path).stem}_limit{max_length}.json"
    )

    # This split-level manifest describes filtering, independent of run budget.
    split_info = dict(info)
    split_info["selected_count"] = len(eligible_rows)
    split_info["selected_indices"] = eligible_indices
    split_info["stratum_counts"] = {
        key: {
            "original": counts["original"],
            "retained": counts["retained"],
        }
        for key, counts in stratum_counts.items()
    }
    save_json(manifest, split_info)

    print(
        f"{Path(path).name}: {len(original_rows)} original, "
        f"{len(excluded_indices)} excluded, "
        f"{len(selected_rows)} selected"
    )
    return selected_rows, info
