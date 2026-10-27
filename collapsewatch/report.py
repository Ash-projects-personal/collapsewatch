"""Text and JSON reports for a check run."""

import json


def text_report(path, measurements, scored):
    """Human-readable report, printed by the CLI."""
    m = measurements
    lines = []
    lines.append("CollapseWatch: %s" % path)
    lines.append("rows: %d" % m["rows"])
    lines.append("")
    lines.append(
        "collapse risk score: %.1f / 100  verdict: %s"
        % (scored["score"], scored["verdict"])
    )
    lines.append("")
    lines.append("measurements (higher means more collapsed):")
    lines.append(
        "  exact duplicates:   %d rows (%.1f%%)"
        % (m["exact_dup_rows"], m["exact_dup_rate"] * 100)
    )
    lines.append(
        "  near duplicates:    %d rows (%.1f%%) at 5-gram Jaccard >= %.2f"
        % (m["near_dup_rows"], m["near_dup_rate"] * 100, m["near_dup_threshold"])
    )
    lines.append(
        "  template clusters:  %d distinct prompt templates" % m["template_clusters"]
    )
    lines.append(
        "  largest template:   %.1f%% of rows" % (m["top1_template_share"] * 100)
    )
    lines.append(
        "  top 3 templates:    %.1f%% of rows" % (m["top3_template_share"] * 100)
    )
    lines.append(
        "  top-50 word cover:  %.1f%% of response tokens"
        % (m["vocab_top50_coverage"] * 100)
    )
    lines.append("  type-token ratio:   %.3f" % m["type_token_ratio"])
    lines.append(
        "  response length:    mean %.1f tokens, median %s, range %d-%d"
        % (
            m["response_length"]["mean"],
            m["response_length"]["median"],
            m["response_length"]["min"],
            m["response_length"]["max"],
        )
    )
    lines.append(
        "  modal length bucket: %.1f%% of rows in %s tokens"
        % (m["length_modal_share"] * 100, m["length_modal_bucket_tokens"])
    )
    lines.append(
        "  answer repetition:  %.3f (mean pairwise response similarity inside templates)"
        % m["answer_repetition"]
    )
    lines.append("")
    lines.append("score breakdown (points out of 100):")
    for key, points in scored["components"].items():
        lines.append("  %-20s %5.1f" % (key, points))
    lines.append("")
    lines.append("worst templates (largest first):")
    for item in m["worst_templates"]:
        sim = item["mean_pairwise_response_similarity"]
        sim_text = "%.3f" % sim if sim is not None else "n/a"
        lines.append(
            '  "%s"  %d rows (%.1f%%), %d distinct answers, answer similarity %s'
            % (
                item["template"],
                item["rows"],
                item["share"] * 100,
                item["distinct_answers"],
                sim_text,
            )
        )
        lines.append("    e.g. %s" % item["example_prompt"])
    return "\n".join(lines) + "\n"


def json_report(path, measurements, scored):
    return {
        "file": path,
        "measurements": measurements,
        "score": scored["score"],
        "score_components": scored["components"],
        "verdict": scored["verdict"],
    }


def write_json_report(out_path, report):
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2))
