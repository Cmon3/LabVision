"""Report rules. Called when the agent calls submit_report (and by helper verify).

    run_verification(ctx, report) -> {"passed": bool, "feedback": [problem, ...]}
"""

from __future__ import annotations

import re

# Benchmarked prompts (from config/bench/RESULTS.md)
BENCHMARKED_PROMPTS = {"round spot", "colonies", "cell"}

# Blocklist: real organism names only + diagnosis (no generic species/organism words)
ORGANISM_BLOCKLIST = [
    "escherichia", "staphylococcus", "streptococcus", "bacillus",
    "pseudomonas", "salmonella", "klebsiella", "enterococcus",
    "proteus", "citrobacter", "serratia", "micrococcus",
    "candida", "aspergillus", "clostridium", "listeria",
    "shigella", "campylobacter", "vibrio", "helicobacter",
]
FORBIDDEN_WORDS = ["diagnosis", "diagnose"]


def _numbers_in_text(text: str) -> list[str]:
    # Find integers/decimals that look like counts/measurements (ignore years, filter out small context?)
    # We check every integer >=0
    return re.findall(r"\b\d+(?:\.\d+)?\b", text or "")


def _countable_wording(text: str, count: int) -> bool:
    """True if text correctly reflects countable range 30-300."""
    t = text.lower()
    # must match the feedback the model gets: "too few" or "low statistical reliability"
    low = "too few" in t or "low statistical reliability" in t
    tntc = "tntc" in text.lower() or "too numerous to count" in text.lower()
    if count < 30:
        return low or tntc  # accept tntc wording for low? spec says "TNTC / too few"
    if count > 300:
        return tntc
    # countable: must NOT say tntc/too few
    return not (tntc and count <= 300) and not (low and 30 <= count <= 300)


def run_verification(ctx: dict, report: dict) -> dict:
    """Verify the report.

    ctx: {
        count: int (measure count with filters applied),
        rows: list[dict] (measured rows),
        qc: dict | None (qc result with is_blurry/glare),
        prompt_used: str | None,
        experimental: bool,
        filter_info: str | None,  # e.g. min_diameter
    }
    report: {summary, count, filters, notes}

    Returns {"passed": bool, "feedback": [str, ...]}
    """
    problems: list[str] = []
    count = ctx.get("count")
    rows = ctx.get("rows") or []
    qc = ctx.get("qc") or {}
    prompt_used = (ctx.get("prompt_used") or "").lower().strip()
    experimental = bool(ctx.get("experimental"))
    # report fields
    summary = str(report.get("summary") or "")
    report_count = report.get("count")
    filters = report.get("filters")
    notes = str(report.get("notes") or "")

    # 1. count equals measure count (with same filters)
    if report_count is None:
        problems.append("missing count: submit_report must include count")
    else:
        try:
            rc = int(report_count)
            if rc != count:
                problems.append(f"count {rc} does not match measured count {count} (with filters {filters})")
        except Exception:
            problems.append(f"count must be an integer, got {report_count!r}")

    # 2. every number in summary appears in measured results
    # allowed numbers are those derived from rows + count
    allowed_numbers: set[str] = set()
    if count is not None:
        allowed_numbers.add(str(int(count)))
    for r in rows:
        for k in ("area_px", "diameter_px", "area_mm2", "diameter_mm"):
            if k in r:
                v = r[k]
                # allow rounded ints and 1-decimal forms
                allowed_numbers.add(str(int(round(v))))
                allowed_numbers.add(f"{v:.1f}")
                allowed_numbers.add(f"{v:.0f}")
                allowed_numbers.add(f"{v:.2f}")
        # also plate/count context
        if "diameter_mm" in r:
            allowed_numbers.add(f"{r['diameter_mm']:.2f}")

    # also allow filter threshold numbers
    if filters:
        nums = _numbers_in_text(str(filters))
        for n in nums:
            allowed_numbers.add(n)

    # check ALL numbers — only measured values, filter threshold, and policy constants 30/300/90 are allowed
    ALLOWED_NON_MEASURED = {"30", "300", "90", "90.0", "90.00"}
    for num in _numbers_in_text(summary):
        if num in ALLOWED_NON_MEASURED:
            continue
        if num not in allowed_numbers:
            problems.append(f"number {num} in summary not found in measured results (no number may come from the LLM)")
            break
    for num in _numbers_in_text(notes):
        if num in ALLOWED_NON_MEASURED:
            continue
        if num not in allowed_numbers:
            problems.append(f"number {num} in notes not found in measured results")
            break

    # 3. QC passed or escalated for re-photo
    is_blurry = bool(qc.get("is_blurry"))
    glare = bool(qc.get("glare"))
    if is_blurry or glare:
        # must either have rephoto escalation or summary notes it
        reason = "blurry" if is_blurry else "glare"
        if "re-photo" not in summary.lower() and "rephoto" not in summary.lower() and "retake" not in summary.lower() and "re-take" not in summary.lower():
            # check if ctx indicates escalated (handled outside, but verification sees qc fail)
            # For now, require summary to acknowledge or report must indicate QC issue
            # ponytail: if QC fails, report must not claim OK — check notes/QC
            problems.append(f"QC failed ({reason}): report must acknowledge re-photo or be escalated (got summary without re-photo wording)")

    # 4. countable range: TNTC / too few wording
    if count is not None:
        combined = summary + " " + notes
        if not _countable_wording(combined, int(count)):
            if int(count) < 30:
                problems.append(f"count {count} is too few (<30): report must say 'too few' or 'low statistical reliability'")
            elif int(count) > 300:
                problems.append(f"count {count} is TNTC (>300): report must say 'TNTC' or 'too numerous to count'")
            else:
                # countable but mentions tntc/too few incorrectly
                if "tntc" in combined.lower() or "too few" in combined.lower():
                    problems.append(f"count {count} is countable (30-300): report must not say TNTC/too few")
                else:
                    problems.append(f"count {count} is countable (30-300): report should not need special wording")

    # 5. prompt benchmarked or experimental
    if prompt_used:
        is_benchmarked = prompt_used in BENCHMARKED_PROMPTS
        if not is_benchmarked and not experimental:
            # check if summary/notes marks experimental
            combined2 = (summary + " " + notes).lower()
            if "experimental" not in combined2:
                problems.append(f"prompt '{prompt_used}' is not benchmarked (round spot/colonies/cell): report must be marked experimental")

    # 6. no organism names and no diagnosis
    lower_summary = summary.lower()
    lower_notes = notes.lower()
    combined_all = lower_summary + " " + lower_notes
    for w in FORBIDDEN_WORDS:
        if w in combined_all:
            problems.append(f"report must not contain '{w}' (no medical/clinical advice)")

    for org in ORGANISM_BLOCKLIST:
        if org in combined_all:
            problems.append(f"report must not contain organism name '{org}' (use morphotype description only)")

    return {"passed": not problems, "feedback": problems}
