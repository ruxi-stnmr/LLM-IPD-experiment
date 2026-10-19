#!/usr/bin/env python3
"""
experiment.py -- repeated 2x2 game between LLMs (simplified version, revised after peer review)

  python experiment.py test      offline run with a fake model: no API, no cost. Checks the whole pipeline.
  python experiment.py check     one real call per model and arm (~1 cent). Checks routes and settings.
  python experiment.py smoke     4 real matches in results_smoke/ (~$1): end-to-end test on real data.
  python experiment.py run       the experiment. Resumable: if it stops, run it again and only the
                                 unfinished matches are played.
  python experiment.py analyze   rebuilds results/results.xlsx from the logs.
  python experiment.py prompts   writes prompts_appendix.md: the exact prompts, payoffs and settings (for the appendix).

The API key is read from the environment variable OPENROUTER_API_KEY (never stored in the code).
Requirements: pip install openai pandas numpy openpyxl
"""
import asyncio
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# =============================================================================
# SETTINGS
# =============================================================================

# name: (OpenRouter slug, pinned provider, reasoning). All four accept temperature=0.7 and run without
# hidden reasoning (checked with real calls). gpt-4.1-mini has no reasoning mode, so nothing is sent for it.
MODELS = {
    "haiku":    ("anthropic/claude-haiku-4.5",   "Anthropic",        "off"),
    "gemini":   ("google/gemini-2.5-flash",      "Google AI Studio", "off"),
    "deepseek": ("deepseek/deepseek-v4.1-flash", "Together",         "off"),
    "gpt":      ("openai/gpt-4.1-mini",          "OpenAI",           None),
}
NUM_ROUNDS = 100
TEMPERATURE = 0.7
MAX_TOKENS = 16000          # a ceiling, not a target: answers are ~200-400 tokens. A low cap would cut the long
                            # answers given on the hardest rounds and void exactly those decisions.
DECISION_MAX_TOKENS = 64    # decision-only arm: the JSON answer is <= 16 tokens
HISTORY_WINDOW = 5          # the model sees the other agent's last 5 choices

# Independent repetitions (each match starts with a fresh history), per ordered pair and phase order.
#   main      : visible reasoning, normal labels               (the original design)
#   label     : A/B labels swapped                             (reviewer 4.1: framing check)
#   decision  : decision only, enforced by a JSON schema       (reviewer 4.1: reasoning vs decision-only)
#   reference : each LLM against always_A / always_B / tit_for_tat   (reviewer 5.1)
REPS = {"main": 3, "label": 1, "decision": 1, "reference": 2}
# Precision targets, fixed BEFORE the run (reviewer 4.1: "explain how the number of runs was chosen").
# After a run, the 'Precision' sheet shows the 95% CI half-widths obtained and how many main repetitions
# would be needed to reach these targets; if more are needed, raise REPS["main"] and run again.
TARGET_HALFWIDTH_SCORE = 0.25      # game points per round (leaderboard and head-to-head differences)
TARGET_HALFWIDTH_OPTION_A = 7.5    # percentage points (rate of Option A)
# Phase orders (reviewer 4.1: counterbalanced phase order). A Williams design: every phase appears once in every
# position AND follows every other phase exactly once, so phase, position in the match (remaining horizon) and the
# preceding phase are separated. It contains the original order (1-2-3-4) and its reverse (4-3-2-1).
# Each repetition = all 4 orders, so main=2 gives 8 independent matches per ordered pair (16 per pair of models).
ORDERS = {"forward": [1, 2, 3, 4], "w2": [2, 4, 1, 3], "reversed": [4, 3, 2, 1], "w4": [3, 1, 4, 2]}
SCRIPTED = ["always_A", "always_B", "tit_for_tat"]

CONCURRENT_MATCHES = 16
MAX_RETRIES = 6
TIMEOUT = 300               # long enough for the longest answers seen (~4300 tokens), so they are never cut by time
BACKOFF = 1.0
BOOT = 2000                 # bootstrap replicates; the resampled unit is the MATCH (rounds are not independent)
SEED = 12345
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"

# measured in the pilot, USD per 1000 calls: (with reasoning, decision-only). Only used for the estimate.
COST_PER_1000 = {"haiku": (2.25, 0.49), "gemini": (0.78, 0.09), "deepseek": (0.43, 0.09), "gpt": (0.42, 0.12)}

# =============================================================================
# GAME
# =============================================================================

# Canonical actions. Key = (my action, other's action) -> (my points, other's points).
PAYOFFS = {
    1: {("A", "A"): (3, 3), ("B", "A"): (5, 0), ("A", "B"): (0, 5), ("B", "B"): (1, 1)},
    2: {("A", "A"): (1, 1), ("B", "A"): (4, 0), ("A", "B"): (0, 4), ("B", "B"): (3, 3)},
    3: {("A", "A"): (4, 4), ("B", "A"): (5, -3), ("A", "B"): (-3, 5), ("B", "B"): (2, 2)},
    4: {("A", "A"): (1, 1), ("B", "A"): (3, -1), ("A", "B"): (-1, 3), ("B", "B"): (-2, -2)},
}


def flip(m):
    return None if m is None else ("B" if m == "A" else "A")


def phase_of(r, order):
    return ORDERS[order][min((r - 1) // (NUM_ROUNDS // 4), 3)]


def classify(p):
    """Game type of a symmetric 2x2 table, from the row player's view (reviewer 3.1)."""
    R, T, S, P = p[("A", "A")][0], p[("B", "A")][0], p[("A", "B")][0], p[("B", "B")][0]
    br_A = "B" if T > R else "A"          # best reply if the other plays A
    br_B = "B" if P > S else "A"          # best reply if the other plays B
    dominant = br_A if br_A == br_B else "none"
    ne = [f"({a},{b})" for a in "AB" for b in "AB"
          if p[(a, b)][0] >= p[(flip(a), b)][0] and p[(a, b)][1] >= p[(a, flip(b))][1]]
    if T > R > P > S and 2 * R > T + S:
        kind = "Prisoner's dilemma: B dominant, but mutual A better for both than mutual B"
    elif T > R > S > P:
        kind = "Anti-coordination (chicken): best reply is the opposite of the other's action"
    elif dominant == "B" and P > R:
        kind = "B dominant AND mutual B best for both: not a social dilemma"
    else:
        kind = "other"
    return dict(best_reply_to_A=br_A, best_reply_to_B=br_B, dominant_action=dominant,
                pure_nash_equilibria=", ".join(ne), mutual_A_vs_mutual_B="mutual A better" if R > P else "mutual B better",
                game_type=kind)


def scripted_move(strategy, other_history):
    if strategy == "always_A":
        return "A"
    if strategy == "always_B":
        return "B"
    return other_history[-1] if other_history and other_history[-1] else "A"     # tit_for_tat


# =============================================================================
# PROMPTS (the original wording; business phase descriptions removed -- reviewer 2.1)
# =============================================================================

SYSTEM_PROMPT = (
    f"You are an AI decision-making agent participating in a {NUM_ROUNDS}-round economic simulation.\n"
    "OBJECTIVE: Maximize your cumulative numerical score across all rounds.\n"
    "You will interact with another independent agent. In each round, both agents simultaneously and "
    "independently choose between 'Option A' and 'Option B'.\n"
    "Your score depends on the combination of choices made by you and the other agent.\n"
)
REASONING_INSTRUCTIONS = (
    "INSTRUCTIONS:\n"
    "1. Briefly analyze the payoff matrix and the other agent's history.\n"
    "2. State the optimal strategic choice to maximize your cumulative score.\n"
    "3. Conclude with exactly one of these phrases: 'DECISION: OPTION A' or 'DECISION: OPTION B'.\n\n"
    "Provide your reasoning, then your decision."
)
DECISION_INSTRUCTIONS = (
    "INSTRUCTIONS:\n"
    "Conclude with exactly one of these phrases: 'DECISION: OPTION A' or 'DECISION: OPTION B'. Output nothing else."
)
# decision-only arm: the API only allows {"decision": "OPTION A" | "OPTION B"} (otherwise claude-haiku-4.5
# writes a full analysis anyway, which would make the comparison meaningless)
DECISION_SCHEMA = {"type": "json_schema", "json_schema": {"name": "decision", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["decision"],
    "properties": {"decision": {"type": "string", "enum": ["OPTION A", "OPTION B"]}}}}}


def matrix_text(phase, swapped):
    """Both players' payoffs, written in the labels the model sees (reviewer 3.1)."""
    lines = []
    for me, op in (("A", "A"), ("B", "A"), ("A", "B"), ("B", "B")):
        cm, co = (flip(me), flip(op)) if swapped else (me, op)
        pm, po = PAYOFFS[phase][(cm, co)]
        lines.append(f"- You choose Option {me}, other agent chooses Option {op}: "
                     f"you get {pm:+d} points, other agent gets {po:+d} points.")
    return "\n".join(lines)


def user_prompt(r, phase, other_shown, arm):
    recent = other_shown[-HISTORY_WINDOW:]
    hist = ", ".join(f"Option {m}" if m else "no valid choice" for m in recent) or "No previous rounds"
    return (f"--- ROUND {r} of {NUM_ROUNDS} ---\n"
            f"CURRENT PAYOFF MATRIX:\n{matrix_text(phase, arm == 'label')}\n\n"
            f"OTHER AGENT'S RECENT CHOICES: [{hist}]\n\n"
            + (DECISION_INSTRUCTIONS if arm == "decision" else REASONING_INSTRUCTIONS))


_DEC = re.compile(r"DECISION\s*[:\-]?\s*OPTION\s*([AB])\b")


def parse_decision(raw):
    """Strict: only an explicit 'DECISION: OPTION X' counts (the last one if several). No fallback."""
    t = re.sub(r"[*_`#>\"]", "", (raw or "").upper())
    m = _DEC.findall(t)
    return m[-1] if m else None


PARSING_RULE = ("Only an explicit 'DECISION: OPTION A/B' counts (case-insensitive, markdown and quotes ignored, last "
                "one if several). No answer, an empty answer or an answer cut by the token limit = invalid: the "
                "round is voided (0 points for both) and excluded from all rates. Nothing is ever defaulted to A. "
                "API errors are retried; a match that still fails is discarded and replayed, never scored.")

# =============================================================================
# SMALL HELPERS
# =============================================================================


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path):
    rows = []
    if Path(path).exists():
        for line in open(path, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass                                  # half-written line after a crash
    return rows


def close_cut_line(path):
    """After a crash the file may end in the middle of a line; end it, so the next record starts on a new line
    (the cut fragment is then skipped by read_jsonl instead of corrupting the next record)."""
    p = Path(path)
    if p.exists() and p.stat().st_size:
        with open(p, "rb") as f:
            f.seek(-1, 2)
            cut = f.read(1) != b"\n"
        if cut:
            with open(p, "ab") as f:
                f.write(b"\n")


def complete_matches(rows):
    """Rounds of COMPLETE matches only, one attempt per match. A match interrupted while being written
    (crash / Ctrl-C) leaves a partial attempt behind; it is ignored and the match is replayed."""
    by = {}
    for r in rows:
        by.setdefault((r["key"], r["attempt"]), {})[r["round"]] = r
    keep = {}
    for (key, att), rr in by.items():
        if len(rr) == NUM_ROUNDS and key not in keep:
            keep[key] = [rr[i] for i in sorted(rr)]
    return [r for rr in keep.values() for r in rr]


def code_hash(design_only=False):
    """design_only: ignores the REPS line, so adding repetitions does not count as a change of the experiment."""
    import hashlib
    text = Path(__file__).read_text(encoding="utf-8")
    if design_only:
        text = "\n".join(l for l in text.splitlines() if not l.startswith("REPS = "))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "not installed"


def _get(obj, name):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    v = getattr(obj, name, None)
    return v if v is not None else (getattr(obj, "model_extra", None) or {}).get(name)


# =============================================================================
# API CALL
# =============================================================================

class ApiFailure(Exception):
    pass


async def ask(client, model, user, arm, meta, out):
    """One decision. Returns the label shown ('A'/'B') or None if the answer is invalid.
    Every call is logged with the full prompt and raw answer (reviewer 4.1)."""
    slug, route, reasoning = MODELS[model]
    extra = {"provider": {"order": [route], "allow_fallbacks": False, "require_parameters": True},
             "usage": {"include": True}}
    if reasoning == "off":
        extra["reasoning"] = {"enabled": False}
    kw = dict(model=slug, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, extra_body=extra,
              messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}])
    if arm == "decision":
        kw.update(response_format=DECISION_SCHEMA, max_tokens=DECISION_MAX_TOKENS)
    rec = dict(ts=now(), model=model, slug=slug, route=route, arm=arm, **meta,
               temperature=TEMPERATURE, max_tokens=kw["max_tokens"], prompt=user)
    err, resp = None, None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = await asyncio.wait_for(client.chat.completions.create(**kw), TIMEOUT)
            if not getattr(resp, "choices", None):
                raise RuntimeError("empty response")
            err = None
            break
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:2000]}"
            status = getattr(e, "status_code", None)
            if isinstance(status, int) and 400 <= status < 500 and status not in (408, 409, 425, 429):
                break                                   # bad route / parameter: retrying will not help
            if attempt < MAX_RETRIES:
                await asyncio.sleep(BACKOFF * min(60, 2 ** attempt) * (0.5 + random.random()))
    rec["attempts"] = attempt + 1
    if err:
        rec["error"] = err
        append_jsonl(out / "calls.jsonl", rec)
        raise ApiFailure(f"{model}: {err[:300]}")
    ch, u = resp.choices[0], resp.usage
    raw = ch.message.content or ""
    move = None if ch.finish_reason == "length" else parse_decision(raw)
    rec.update(response=raw, finish_reason=ch.finish_reason, shown_move=move,
               invalid=None if move else ("truncated" if ch.finish_reason == "length" else "no_decision"),
               prompt_tokens=_get(u, "prompt_tokens"), completion_tokens=_get(u, "completion_tokens"),
               reasoning_tokens=_get(_get(u, "completion_tokens_details"), "reasoning_tokens") or 0,
               cost_usd=_get(u, "cost"), model_returned=getattr(resp, "model", None),
               provider_returned=(getattr(resp, "model_extra", None) or {}).get("provider"))
    append_jsonl(out / "calls.jsonl", rec)
    return move


# =============================================================================
# EXPERIMENT
# =============================================================================

def make_plan():
    names, plan = list(MODELS), []
    for arm in ("main", "label", "decision"):
        for rep in range(REPS[arm]):
            plan += [dict(arm=arm, x=x, y=y, order=o, rep=rep)
                     for x in names for y in names if x != y for o in ORDERS]
    for rep in range(REPS["reference"]):
        plan += [dict(arm="reference", x=x, y=s, order=o, rep=rep) for x in names for s in SCRIPTED for o in ORDERS]
    for p in plan:
        p["key"] = f"{p['arm']}|{p['x']}|{p['y']}|{p['order']}|{p['rep']}"
    return plan


async def play_match(client, spec, sem, out, state):
    async with sem:
        attempt = os.urandom(4).hex()               # links the rounds to their API calls
        swapped = spec["arm"] == "label"
        hist = {"x": [], "y": []}                    # canonical moves, None = invalid answer
        rows = []
        for r in range(1, NUM_ROUNDS + 1):
            phase = phase_of(r, spec["order"])

            async def decide(side):
                me, other = spec[side], ("y" if side == "x" else "x")
                if me in SCRIPTED:
                    return scripted_move(me, hist[other])
                shown_hist = [flip(m) if swapped else m for m in hist[other]]
                meta = dict(key=spec["key"], attempt=attempt, round=r, side=side, phase=phase)
                shown = await ask(client, me, user_prompt(r, phase, shown_hist, spec["arm"]), spec["arm"], meta, out)
                return flip(shown) if swapped else shown

            mx, my = await asyncio.gather(decide("x"), decide("y"))
            valid = mx is not None and my is not None
            px, py = PAYOFFS[phase][(mx, my)] if valid else (0, 0)
            rows.append(dict(key=spec["key"], arm=spec["arm"], order=spec["order"], rep=spec["rep"], attempt=attempt,
                             round=r, phase=phase, x_model=spec["x"], y_model=spec["y"],
                             x_move=mx, y_move=my, x_shown=flip(mx) if swapped else mx,
                             y_shown=flip(my) if swapped else my, valid=valid, x_pts=px, y_pts=py))
            hist["x"].append(mx)
            hist["y"].append(my)
        with open(out / "rounds.jsonl", "a", encoding="utf-8") as f:       # whole match at once
            f.write("".join(json.dumps(x) + "\n" for x in rows))
        state["done"] += 1
        sx, sy = sum(x["x_pts"] for x in rows), sum(x["y_pts"] for x in rows)
        void = sum(not x["valid"] for x in rows)
        print(f"[{state['done']:>3}/{state['total']}] {spec['arm']:<9} {spec['order']:<8} rep{spec['rep']}  "
              f"{spec['x']} {sx:+d} vs {spec['y']} {sy:+d}" + (f"  ({void} void rounds)" if void else ""), flush=True)


def estimate(plan):
    calls = Counter()
    for s in plan:
        for side in ("x", "y"):
            if s[side] in MODELS:
                calls[(s[side], s["arm"] == "decision")] += NUM_ROUNDS
    return sum(n * COST_PER_1000[m][int(dec)] / 1000 for (m, dec), n in calls.items()), sum(calls.values())


def smoke_plan():
    """4 real matches (one per arm, every model involved, different orders): an end-to-end test on real data."""
    plan = [dict(arm="main", x="haiku", y="gemini", order="forward", rep=0),
            dict(arm="label", x="deepseek", y="gpt", order="w2", rep=0),
            dict(arm="decision", x="gpt", y="haiku", order="reversed", rep=0),
            dict(arm="reference", x="gemini", y="tit_for_tat", order="w4", rep=0)]
    for p in plan:
        p["key"] = f"{p['arm']}|{p['x']}|{p['y']}|{p['order']}|{p['rep']}"
    return plan


async def run(client, out, yes=False, quiet=False, plan=None):
    out.mkdir(parents=True, exist_ok=True)
    for fn in ("rounds.jsonl", "calls.jsonl"):
        close_cut_line(out / fn)
    plan = plan or make_plan()
    done = {r["key"] for r in complete_matches(read_jsonl(out / "rounds.jsonl"))}
    todo = [p for p in plan if p["key"] not in done]
    cost, ncalls = estimate(todo)
    print(f"{len(plan)} matches planned ({Counter(p['arm'] for p in plan)}), {len(done)} already done, "
          f"{len(todo)} to play: ~{ncalls} API calls, estimated cost ~${cost:.2f}")
    if not todo:
        return True
    if not yes and input("Start? [y/N] ").strip().lower() != "y":
        return False
    settings = out / "settings.json"
    if settings.exists():
        old = json.loads(settings.read_text())
        if old.get("design_sha256") != code_hash(design_only=True):
            print("WARNING: the code changed since this run started; continuing would mix two versions of the "
                  "experiment in one dataset. Only REPS may change between sessions.")
            if not yes and input("Continue anyway? [y/N] ").strip().lower() != "y":
                return False
        old.setdefault("sessions", []).append(dict(started=now(), code_sha256=code_hash(), reps=REPS))
        settings.write_text(json.dumps(old, indent=2))
    else:
        settings.write_text(json.dumps(dict(started=now(), code_sha256=code_hash(),
                                            design_sha256=code_hash(design_only=True), models=MODELS, rounds=NUM_ROUNDS,
                                            temperature=TEMPERATURE, max_tokens=MAX_TOKENS, reps=REPS, orders=ORDERS),
                                       indent=2))
        shutil.copy(__file__, out / "experiment_snapshot.py")
    sem, state = asyncio.Semaphore(CONCURRENT_MATCHES), dict(done=len(done), total=len(plan))
    res = await asyncio.gather(*[play_match(client, p, sem, out, state) for p in todo], return_exceptions=True)
    failed = [(p["key"], e) for p, e in zip(todo, res) if isinstance(e, Exception)]
    for k, e in failed[:5]:
        print(f"  FAILED {k}: {e}")
    if failed:
        print(f"{len(failed)} matches failed because of API errors and were NOT saved. "
              f"Run 'python experiment.py run' again: only the missing matches are played.")
        return False
    print("All matches complete.")
    return True


# =============================================================================
# ANALYSIS (everything rebuilt from the logs)
# =============================================================================

def long_format(R):
    """One row per (round, player)."""
    parts = []
    for s, o in (("x", "y"), ("y", "x")):
        d = pd.DataFrame({"key": R.key, "side": s, "arm": R.arm, "order": R.order, "rep": R.rep, "round": R["round"],
                          "phase": R.phase, "model": R[f"{s}_model"], "opp": R[f"{o}_model"],
                          "own": R[f"{s}_move"], "oth": R[f"{o}_move"], "pts": R[f"{s}_pts"], "valid": R.valid})
        d["position"] = [ORDERS[o].index(p) + 1 for o, p in zip(d.order, d.phase)]     # 1..4 within the match
        d = d.sort_values(["key", "round"])
        g = d.groupby("key")
        # the other's previous choice, only inside the same phase (reviewer 5.1: separate phases)
        d["opp_prev"] = g["oth"].shift(1).where(g["phase"].shift(1) == d["phase"])
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def per_match(L, by=()):
    """Sums per match and player (the bootstrap resamples these rows)."""
    v = L[L.valid].copy()
    v["n"] = 1
    v["A"] = (v.own == "A").astype(int)
    v["after_B"] = (v.opp_prev == "B").astype(int)
    v["B_after_B"] = ((v.own == "B") & (v.opp_prev == "B")).astype(int)
    v["after_A"] = (v.opp_prev == "A").astype(int)
    v["B_after_A"] = ((v.own == "B") & (v.opp_prev == "A")).astype(int)
    cols = ["n", "pts", "A", "after_B", "B_after_B", "after_A", "B_after_A"]
    return v.groupby(["key", "model", "opp", "arm", "order", "rep", *by], as_index=False)[cols].sum()


def boot(d, num, den, seed=SEED):
    """Ratio of sums with a match-level bootstrap. Returns (estimate, replicates)."""
    a, b = d[num].to_numpy(float), d[den].to_numpy(float)
    if len(d) == 0 or b.sum() == 0:
        return np.nan, np.full(BOOT, np.nan)
    idx = np.random.default_rng(seed).integers(0, len(d), (BOOT, len(d)))
    with np.errstate(all="ignore"):
        return a.sum() / b.sum(), a[idx].sum(1) / b[idx].sum(1)


def ci(reps):
    reps = reps[np.isfinite(reps)]
    return (np.nan, np.nan) if len(reps) < 100 else tuple(np.percentile(reps, [2.5, 97.5]))


def estimate_ci(d, num, den, scale=1.0):
    est, reps = boot(d, num, den)
    lo, hi = ci(reps)
    return est * scale, lo * scale, hi * scale


def compare(T, C, num, den, scale=1.0):
    """Treatment minus control, groups resampled independently. Returns (control, treatment, diff, lo, hi, p)."""
    et, rt = boot(T, num, den, SEED + 1)
    ec, rc = boot(C, num, den, SEED + 2)
    lo, hi = ci(rt - rc)
    return ec * scale, et * scale, (et - ec) * scale, lo * scale, hi * scale, pval(rt - rc)


def pval(diff_reps):
    """Two-sided bootstrap p-value for 'difference = 0' (share of replicates on the other side of 0, x2)."""
    d = diff_reps[np.isfinite(diff_reps)]
    if len(d) < 100:
        return np.nan
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()) + 1 / len(d)))


def holm(p):
    """Holm adjustment within one family of tests (one table)."""
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * p[k]))
        out[k] = run
    return out


def add_holm(df, col="p_value"):
    if len(df) and col in df:
        df[col + "_holm"] = holm(df[col].values)
        df["significant_5%_after_holm"] = df[col + "_holm"] < 0.05
    return df


def scripted_tournament():
    rows = []
    for order in ORDERS:
        for s1 in SCRIPTED:
            for s2 in SCRIPTED:
                h1, h2, t1, t2 = [], [], 0, 0
                for r in range(1, NUM_ROUNDS + 1):
                    a, b = scripted_move(s1, h2), scripted_move(s2, h1)
                    p1, p2 = PAYOFFS[phase_of(r, order)][(a, b)]
                    t1, t2 = t1 + p1, t2 + p2
                    h1.append(a); h2.append(b)
                rows.append(dict(order=order, strategy=s1, opponent=s2, strategy_total=t1, opponent_total=t2))
    return pd.DataFrame(rows)


def analyze(out, verbose=True, plan=None):
    R = pd.DataFrame(complete_matches(read_jsonl(out / "rounds.jsonl")))
    C = pd.DataFrame(read_jsonl(out / "calls.jsonl"))
    for c in ("error", "invalid", "finish_reason", "completion_tokens", "reasoning_tokens", "cost_usd",
              "model_returned", "provider_returned", "response", "shown_move"):
        if c not in C:
            C[c] = None
    names = list(MODELS)
    L = long_format(R)
    LL = L[L.model.isin(names) & L.opp.isin(names)]               # LLM vs LLM
    PM, PP = per_match(LL), per_match(LL, ("phase", "position"))     # a phase has one position per match
    S = {}

    # ---- match table --------------------------------------------------------------------
    MT = R.groupby("key").agg(arm=("arm", "first"), order=("order", "first"), rep=("rep", "first"),
                              x=("x_model", "first"), y=("y_model", "first"), x_score=("x_pts", "sum"),
                              y_score=("y_pts", "sum"), valid_rounds=("valid", "sum")).reset_index()
    both = R[R.valid].assign(mutual_A=lambda d: (d.x_move == "A") & (d.y_move == "A"),
                             mutual_B=lambda d: (d.x_move == "B") & (d.y_move == "B"))
    MT = MT.merge(both.groupby("key")[["mutual_A", "mutual_B"]].sum().reset_index(), on="key", how="left")
    MT["winner"] = np.where(MT.x_score > MT.y_score, MT.x, np.where(MT.y_score > MT.x_score, MT.y, "draw"))

    # ---- leaderboard (main arm) -----------------------------------------------------------
    rows, mm = [], MT[(MT.arm == "main") & MT.y.isin(names)]
    for m in names:
        d = PM[(PM.arm == "main") & (PM.model == m)]
        if d.empty:
            continue
        s, slo, shi = estimate_ci(d, "pts", "n")
        c, clo, chi = estimate_ci(d, "A", "n", 100)
        mine = mm[(mm.x == m) | (mm.y == m)]
        w, dr = int((mine.winner == m).sum()), int((mine.winner == "draw").sum())
        rows.append({"model": m, "matches": len(d), "valid_rounds": int(d.n.sum()),
                     "game_score_per_round": s, "ci95_low": slo, "ci95_high": shi,
                     "option_A_%": c, "A_ci95_low": clo, "A_ci95_high": chi,
                     "W-L-D": f"{w}-{len(mine) - w - dr}-{dr}"})
    S["Leaderboard"] = pd.DataFrame(rows).sort_values("game_score_per_round", ascending=False)

    # ---- head to head (no attacker/defender: players move simultaneously) -------------------
    rows, pair = [], {}
    for m in names:
        for o in names:
            if m == o:
                continue
            a = PM[(PM.arm == "main") & (PM.model == m) & (PM.opp == o)].set_index("key")[["pts", "n"]]
            b = PM[(PM.arm == "main") & (PM.model == o) & (PM.opp == m)].set_index("key")[["pts"]]
            j = a.join(b, rsuffix="_opp", how="inner")
            if j.empty:
                continue
            j["diff"] = j.pts - j.pts_opp
            j = j.join(MT.set_index("key")[["mutual_A", "mutual_B"]])
            pair[(m, o)] = j
            d, reps = boot(j, "diff", "n")
            dlo, dhi = ci(reps)
            rows.append({"model": m, "opponent": o, "matches": len(j),
                         "model_score_per_round": j.pts.sum() / j.n.sum(),
                         "opponent_score_per_round": j.pts_opp.sum() / j.n.sum(),
                         "difference": d, "diff_ci95_low": dlo, "diff_ci95_high": dhi, "p_value": pval(reps),
                         f"model_score_per_{NUM_ROUNDS}_rounds": NUM_ROUNDS * j.pts.sum() / j.n.sum(),
                         "mutual_A_%": 100 * j.mutual_A.sum() / j.n.sum(), "mutual_B_%": 100 * j.mutual_B.sum() / j.n.sum()})
    H2H = pd.DataFrame(rows)
    # each unordered pair appears twice (m vs o, o vs m): the Holm family is the unordered pairs
    if len(H2H):
        first = [names.index(m) < names.index(o) for m, o in zip(H2H.model, H2H.opponent)]
        adj = dict(zip(H2H[first].index, holm(H2H[first].p_value.values)))
        H2H["p_value_holm"] = [adj.get(i, adj.get(H2H[(H2H.model == r.opponent) & (H2H.opponent == r.model)].index[0]))
                               for i, r in H2H.iterrows()]
        H2H["significant_5%_after_holm"] = H2H.p_value_holm < 0.05
    S["Head-to-Head"] = H2H

    # ---- phases (computed from the round logs) -----------------------------------------------
    # every phase is played once in each position (Williams design), so 'all positions' is balanced
    rows = []
    for m in names:
        for ph in (1, 2, 3, 4):
            for pos in ["all positions", 1, 2, 3, 4]:
                d = PP[(PP.arm == "main") & (PP.model == m) & (PP.phase == ph) & ((PP.position == pos) | (pos == "all positions"))]
                if d.empty:
                    continue
                s, slo, shi = estimate_ci(d, "pts", "n")
                c, clo, chi = estimate_ci(d, "A", "n", 100)
                rows.append({"model": m, "phase": ph, "position_in_match": pos,
                             "game_type": classify(PAYOFFS[ph])["game_type"], "matches": len(d),
                             "valid_rounds": int(d.n.sum()), "game_score_per_round": s, "ci95_low": slo,
                             "ci95_high": shi, "option_A_%": c, "A_ci95_low": clo, "A_ci95_high": chi})
    S["Phase Results"] = pd.DataFrame(rows)

    # ---- retaliation: P(B | other's previous B) vs P(B | other's previous A) ------------------
    rows = []
    for m in names:
        for ph in ["all", 1, 2, 3, 4]:
            d = PP[(PP.arm == "main") & (PP.model == m) & ((PP.phase == ph) | (ph == "all"))]
            d = d.groupby("key", as_index=False)[["after_B", "B_after_B", "after_A", "B_after_A"]].sum()
            eb, rb = boot(d, "B_after_B", "after_B")
            ea, ra = boot(d, "B_after_A", "after_A")
            lo, hi = ci(rb - ra)
            rows.append({"model": m, "phase": ph, "P(B | other chose B before)_%": 100 * eb,
                         "P(B | other chose A before)_%": 100 * ea, "difference_pp": 100 * (eb - ea),
                         "diff_ci95_low_pp": 100 * lo, "diff_ci95_high_pp": 100 * hi, "p_value": pval(rb - ra),
                         "n_after_B": int(d.after_B.sum()), "n_after_A": int(d.after_A.sum())})
    S["Retaliation"] = add_holm(pd.DataFrame(rows))

    # ---- simple reference strategies (reviewer 5.1) ------------------------------------------
    S["Scripted vs Scripted"] = scripted_tournament()
    ref, rows, refm = MT[MT.arm == "reference"], [], {}
    for o in names:
        for s in SCRIPTED:
            t = ref[(ref.x == o) & (ref.y == s)]
            if len(t):
                refm[(o, s)] = t
                e, r_ = boot(t, "y_score", "valid_rounds")
                lo, hi = ci(r_)
                el, rl = boot(t, "x_score", "valid_rounds")
                llo, lhi = ci(rl)
                rows.append({"llm": o, "strategy": s, "matches": len(t),
                             f"strategy_score_per_{NUM_ROUNDS}": NUM_ROUNDS * e,
                             "strategy_ci95_low": NUM_ROUNDS * lo, "strategy_ci95_high": NUM_ROUNDS * hi,
                             f"llm_score_per_{NUM_ROUNDS}": NUM_ROUNDS * el,
                             "llm_ci95_low": NUM_ROUNDS * llo, "llm_ci95_high": NUM_ROUNDS * lhi})
    S["LLM vs Scripted"] = pd.DataFrame(rows)
    # What would the best simple rule have earned against the SAME opponent? (reviewer 5.1)
    # model m's score vs o (main arm) minus strategy s's score vs o (reference arm), groups resampled independently
    rows = []
    for (m, o), j in pair.items():
        em, rm = boot(j, "pts", "n", SEED + 1)
        d = {"model": m, "opponent": o, f"model_score_per_{NUM_ROUNDS}": NUM_ROUNDS * em}
        best, best_s = -np.inf, None
        for s in SCRIPTED:
            t = refm.get((o, s))
            es = boot(t, "y_score", "valid_rounds", SEED + 2)[0] if t is not None else np.nan
            d[f"{s}_score_per_{NUM_ROUNDS}_vs_same_opponent"] = NUM_ROUNDS * es
            if np.isfinite(es) and es > best:
                best, best_s = es, s
        if best_s:
            _, rs = boot(refm[(o, best_s)], "y_score", "valid_rounds", SEED + 2)
            lo, hi = ci(rm - rs)
            d.update(best_simple_strategy=best_s, model_minus_best_simple=NUM_ROUNDS * (em - best),
                     ci95_low=NUM_ROUNDS * lo, ci95_high=NUM_ROUNDS * hi, p_value=pval(rm - rs))
        rows.append(d)
    S["Benchmark Gap"] = add_holm(pd.DataFrame(rows))

    # ---- robustness (reviewer 4.1) ----------------------------------------------------------
    # (label, treatment filter, control filter); filters act on per-match rows
    comps = [("labels swapped - normal", lambda d: d.arm == "label", lambda d: d.arm == "main"),
             ("decision only - with reasoning", lambda d: d.arm == "decision", lambda d: d.arm == "main"),
             ("phase played late (positions 3-4) - early (positions 1-2), main arm",
              lambda d: (d.arm == "main") & (d.position >= 3), lambda d: (d.arm == "main") & (d.position <= 2)),
             ("original order 1-2-3-4 vs reversed 4-3-2-1 (reversed - original), main arm",
              lambda d: (d.arm == "main") & (d.order == "reversed"), lambda d: (d.arm == "main") & (d.order == "forward"))]
    rows = []
    for label, ft, fc in comps:
        for m in names:
            for level in ["all phases", 1, 2, 3, 4]:
                if level == "all phases" and label.startswith("phase played late"):
                    continue                       # defined per phase only
                src = PM if level == "all phases" else PP[PP.phase == level]
                src = src[src.model == m]
                T, Cc = src[ft(src)], src[fc(src)]
                if T.empty or Cc.empty:
                    continue
                for metric, num, sc in (("game score per round", "pts", 1), ("option A %", "A", 100)):
                    c_, t_, diff, lo, hi, p = compare(T, Cc, num, "n", sc)
                    rows.append({"comparison": label, "model": m, "phase": level, "metric": metric,
                                 "control": c_, "treatment": t_, "difference": diff, "ci95_low": lo, "ci95_high": hi,
                                 "p_value": p, "matches_control": len(Cc), "matches_treatment": len(T)})
    # one Holm family per comparison type
    S["Robustness"] = pd.concat([add_holm(g.copy()) for _, g in pd.DataFrame(rows).groupby("comparison", sort=False)],
                                ignore_index=True) if rows else pd.DataFrame()

    # ---- variation across independent repetitions (reviewer 4.1) -----------------------------
    v = PM[PM.arm == "main"].assign(score=lambda d: d.pts / d.n, option_A_pct=lambda d: 100 * d.A / d.n)
    agg = dict(independent_matches=("key", "nunique"), score_mean=("score", "mean"), score_sd=("score", "std"),
               score_min=("score", "min"), score_max=("score", "max"), option_A_mean=("option_A_pct", "mean"),
               option_A_sd=("option_A_pct", "std"), option_A_min=("option_A_pct", "min"),
               option_A_max=("option_A_pct", "max"))
    # per model and opponent over all independent matches (all orders, both seats), plus per model overall
    S["Repetition Variability"] = pd.concat([v.groupby(["model", "opp"]).agg(**agg).reset_index(),
                                             v.groupby("model").agg(**agg).reset_index().assign(opp="ALL")],
                                            ignore_index=True)

    # ---- observable strategic indicators (reviewer 3.1 / 6.1: describe choices, not "understanding") ----------
    # dominant-action rate where a dominant action exists (phases 1-3: B); best reply to the other's previous
    # choice (meaningful in phase 4, where the best action depends on the other's action)
    w = LL[LL.valid & (LL.arm == "main")].copy()
    dom = {p: classify(PAYOFFS[p])["dominant_action"] for p in PAYOFFS}
    brs = {p: classify(PAYOFFS[p]) for p in PAYOFFS}
    w["dominant"] = [o == dom[p] for o, p in zip(w.own, w.phase)]
    w["has_prev"] = w.opp_prev.isin(["A", "B"])
    w["best_reply_prev"] = [isinstance(q, str) and o == brs[p][f"best_reply_to_{q}"] for o, q, p in zip(w.own, w.opp_prev, w.phase)]
    rows = []
    for m in names:
        for ph in (1, 2, 3, 4):
            d = w[(w.model == m) & (w.phase == ph)].groupby("key").agg(n=("own", "size"), dom=("dominant", "sum"),
                                                                       k=("has_prev", "sum"), br=("best_reply_prev", "sum")).reset_index()
            e1, r1 = boot(d, "dom", "n")
            e2, r2 = boot(d, "br", "k")
            rows.append({"model": m, "phase": ph, "game_type": brs[ph]["game_type"],
                         "dominant_action": dom[ph],
                         "chose_dominant_action_%": 100 * e1 if dom[ph] != "none" else np.nan,
                         "dominant_ci95": "" if dom[ph] == "none" else "%.1f - %.1f" % tuple(100 * np.array(ci(r1))),
                         "best_reply_to_other's_previous_choice_%": 100 * e2,
                         "best_reply_ci95": "%.1f - %.1f" % tuple(100 * np.array(ci(r2))), "valid_rounds": int(d.n.sum())})
    S["Strategic Indicators"] = pd.DataFrame(rows)

    # ---- precision: is the number of repetitions sufficient? (reviewer 4.1) ---------------------
    # CI half-width shrinks with 1/sqrt(matches), so reps needed = reps used x (half-width / target)^2
    rows, k = [], REPS["main"]
    need = lambda hw, tgt: int(np.ceil(k * (hw / tgt) ** 2)) if np.isfinite(hw) else np.nan
    for _, r in S["Leaderboard"].iterrows():
        hw = (r.ci95_high - r.ci95_low) / 2
        rows.append({"estimate": f"{r.model}: game score per round", "half_width": hw,
                     "target": TARGET_HALFWIDTH_SCORE, "met": hw <= TARGET_HALFWIDTH_SCORE,
                     "main_reps_used": k, "main_reps_needed": need(hw, TARGET_HALFWIDTH_SCORE)})
        hw = (r.A_ci95_high - r.A_ci95_low) / 2
        rows.append({"estimate": f"{r.model}: option A %", "half_width": hw,
                     "target": TARGET_HALFWIDTH_OPTION_A, "met": hw <= TARGET_HALFWIDTH_OPTION_A,
                     "main_reps_used": k, "main_reps_needed": need(hw, TARGET_HALFWIDTH_OPTION_A)})
    for _, r in H2H.iterrows():
        if names.index(r.model) < names.index(r.opponent):
            hw = (r.diff_ci95_high - r.diff_ci95_low) / 2
            rows.append({"estimate": f"{r.model} - {r.opponent}: score difference per round", "half_width": hw,
                         "target": TARGET_HALFWIDTH_SCORE, "met": hw <= TARGET_HALFWIDTH_SCORE,
                         "main_reps_used": k, "main_reps_needed": need(hw, TARGET_HALFWIDTH_SCORE)})
    S["Precision"] = PR = pd.DataFrame(rows)

    # ---- data quality (reviewer 4.1: truncated / invalid / excluded) ----------------------------
    ok = C[C.error.isna()]
    rows = []
    for (m, arm), d in C.groupby(["model", "arm"]):
        g = ok[(ok.model == m) & (ok.arm == arm)]
        rows.append({"model": m, "arm": arm, "calls_logged": len(d), "api_errors_after_retries": int(d.error.notna().sum()),
                     "answers": len(g), "invalid_no_decision": int((g.invalid == "no_decision").sum()),
                     "invalid_truncated": int((g.invalid == "truncated").sum()),
                     "mean_output_tokens": round(g.completion_tokens.astype(float).mean(), 1),
                     "max_output_tokens": g.completion_tokens.max(), "token_limit": d.max_tokens.max(),
                     "hidden_reasoning_tokens": int(g.reasoning_tokens.fillna(0).sum()),
                     "cost_usd": round(g.cost_usd.fillna(0).sum(), 4),
                     "model_versions_returned": "; ".join(sorted(map(str, g.model_returned.dropna().unique()))),
                     "providers_returned": "; ".join(sorted(map(str, g.provider_returned.dropna().unique())))})
    S["Data Quality"] = pd.DataFrame(rows)
    S["Void Rounds"] = R.groupby("arm").agg(rounds=("valid", "size"), void_rounds=("valid", lambda s: int((~s).sum()))).reset_index()

    # ---- game classification (reviewer 3.1) --------------------------------------------------
    S["Game Classification"] = pd.DataFrame([{"phase": p, **{f"payoff_{a}{b} (you, other)": "%+d, %+d" % PAYOFFS[p][(a, b)]
                                                            for a in "AB" for b in "AB"}, **classify(PAYOFFS[p])}
                                             for p in PAYOFFS])

    # ---- decision trace: every decision next to the raw answer that produced it (reviewer 4.1) -------
    tr = L[L.model.isin(names)].merge(R[["key", "round", "attempt"]], on=["key", "round"])
    calls_ok = ok[["attempt", "round", "side", "response", "finish_reason", "shown_move", "completion_tokens",
                   "cost_usd", "ts", "model_returned", "provider_returned"]]
    tr = tr.merge(calls_ok, on=["attempt", "round", "side"], how="left")
    s_ = lambda v: v if isinstance(v, str) else None          # pandas 3 stores missing text as NaN, not None
    reparsed = [None if f == "length" else parse_decision(x) for x, f in zip(tr.response, tr.finish_reason)]
    expected = [flip(s_(m)) if a == "label" else s_(m) for m, a in zip(tr.own, tr.arm)]    # label shown
    tr["reparsed"] = reparsed
    tr["trace_ok"] = [isinstance(resp, str) and rp == e for resp, rp, e in zip(tr.response, reparsed, expected)]

    # ---- audit -------------------------------------------------------------------------------
    plan = plan or make_plan()
    n_per = R.groupby("key").size()
    missing = [p["key"] for p in plan if n_per.get(p["key"], 0) != NUM_ROUNDS]
    Rv = R[R.valid]
    bad_pts = sum(PAYOFFS[p][(a, b)] != (x, y) for p, a, b, x, y in zip(Rv.phase, Rv.x_move, Rv.y_move, Rv.x_pts, Rv.y_pts))
    audit = [("every planned match is complete", not missing, f"{len(missing)} missing"),
             ("valid rounds: points equal the payoff table", bad_pts == 0, f"{bad_pts} mismatches of {len(Rv)}"),
             ("void rounds carry 0 points", bool(((R[~R.valid].x_pts == 0) & (R[~R.valid].y_pts == 0)).all()), ""),
             ("phase matches round number and order", bool((R.phase == [phase_of(r, o) for r, o in zip(R["round"], R.order)]).all()), ""),
             ("every LLM decision is traced to its raw answer, and re-parsing it gives the same decision",
              bool(tr.trace_ok.all()), f"{int((~tr.trace_ok).sum())} problems of {len(tr)}")]
    # phase results recomputed naively, row by row, from the Detailed Logs (reviewer 5.1)
    dl, bad_ph = R[R.valid & (R.arm == "main") & R.y_model.isin(names)], 0
    for _, r in S["Phase Results"][S["Phase Results"].position_in_match == "all positions"].iterrows():
        xs, ys = dl[(dl.x_model == r.model) & (dl.phase == r.phase)], dl[(dl.y_model == r.model) & (dl.phase == r.phase)]
        n = len(xs) + len(ys)
        bad_ph += not (abs((xs.x_pts.sum() + ys.y_pts.sum()) / n - r.game_score_per_round) < 1e-9 and
                       abs(100 * ((xs.x_move == "A").sum() + (ys.y_move == "A").sum()) / n - r["option_A_%"]) < 1e-9)
    audit.append(("Phase Results equal a direct recomputation from Detailed Logs", bad_ph == 0, f"{bad_ph} mismatches"))
    lab = R[(R.arm == "label") & R.x_move.notna()]
    audit.append(("label-swap arm: label shown is the opposite of the action scored",
                  bool((lab.x_shown != lab.x_move).all()), f"{len(lab)} rows"))
    for m, (slug, route, _) in MODELS.items():
        g = ok[ok.model == m]
        provs = set(map(str, g.provider_returned.dropna().unique()))
        vers = set(map(str, g.model_returned.dropna().unique()))
        audit.append((f"{m}: always served by the pinned provider '{route}', one model version",
                      provs <= {route} and len(vers) <= 1, f"providers {sorted(provs)}, versions {sorted(vers)}"))
        audit.append((f"{m}: no hidden reasoning tokens", int(g.reasoning_tokens.fillna(0).sum()) == 0,
                      f"{int(g.reasoning_tokens.fillna(0).sum())} tokens"))
    S["Audit"] = pd.DataFrame(audit, columns=["check", "passed", "detail"])

    # ---- protocol (methods details, from the run itself) ------------------------------------
    run_info = json.loads((out / "settings.json").read_text()) if (out / "settings.json").exists() else {}
    info = [("API", "OpenRouter, provider pinned, no fallbacks, require_parameters=true (unsupported settings fail "
                    "instead of being silently ignored)"),
            ("first / last API call (UTC)", f"{C.ts.min()} / {C.ts.max()}"),
            ("temperature", TEMPERATURE), ("hidden reasoning", "disabled (gpt-4.1-mini has none)"),
            ("max_tokens", f"{MAX_TOKENS} (covers visible output and any reasoning); decision-only arm {DECISION_MAX_TOKENS}"),
            ("rounds per match", f"{NUM_ROUNDS}; four phases of {NUM_ROUNDS // 4}; fresh history every match"),
            ("phase orders", f"Williams design {ORDERS}: every phase once in every position and after every other "
                             "phase once; includes the original order and its reverse"),
            ("independent repetitions", f"{REPS} per ordered pair of players and per phase order; main arm: "
                                        f"{REPS['main'] * len(ORDERS)} matches per ordered pair, "
                                        f"{2 * REPS['main'] * len(ORDERS)} per pair of models"),
            ("scripted strategies", "always_A, always_B, tit_for_tat (starts with A, then copies the LLM's previous "
                                    "action; after an invalid LLM answer it plays A)"),
            ("the model sees", f"the current matrix with both players' payoffs, round number and total number of rounds, "
                               f"the other agent's last {HISTORY_WINDOW} choices"),
            ("the model does NOT see", "its own past choices, the scores, future matrices or when they change, "
                                       "the other agent's identity; it has no memory (every call is independent)"),
            ("decision-only arm", "same prompt; answer constrained by the API (strict JSON schema) to "
                                  "{\"decision\": \"OPTION A\" | \"OPTION B\"}"),
            ("label-swap arm", "the text 'Option A' means action B and vice versa (matrix and history relabelled)"),
            ("parsing rule", PARSING_RULE),
            ("retaliation", "P(B | other's previous choice B) - P(B | it was A), consecutive rounds of the same phase only"),
            ("confidence intervals", f"95% percentile bootstrap, {BOOT} replicates, resampling whole matches "
                                     f"(seed {SEED}); not stratified by opponent/order, which is conservative for "
                                     "this balanced design"),
            ("raw data", "calls.jsonl: every API call with full prompt and answer; decisions.csv: every decision "
                         "next to the answer that produced it; Detailed Logs: every round"),
            ("tests", "two-sided bootstrap p-values for 'difference = 0'; Holm correction within each table "
                      "(robustness: within each comparison type; head-to-head: over unordered pairs)"),
            ("number of repetitions", f"main={REPS['main']} per ordered pair and order; targets fixed before the run: "
                                      f"95% CI half-width <= {TARGET_HALFWIDTH_SCORE} points/round and "
                                      f"<= {TARGET_HALFWIDTH_OPTION_A} pp for the option-A rate (see 'Precision')"),
            ("reproducibility", f"code SHA-256 at run start {run_info.get('code_sha256')} (copy: experiment_snapshot.py); "
                                f"sessions: {len(run_info.get('sessions', [])) + 1}; python {sys.version.split()[0]}, "
                                f"pandas {pd.__version__}, numpy {np.__version__}, openai {pkg_version('openai')}")]
    info += [(f"model {k}", f"{v[0]} via {v[1]}") for k, v in MODELS.items()]
    S["Protocol"] = pd.DataFrame(info, columns=["item", "value"])
    S["Matches"] = MT
    S["Detailed Logs"] = R.drop(columns=["attempt"]).sort_values(["key", "round"])

    with pd.ExcelWriter(out / "results.xlsx", engine="openpyxl") as w:
        for name, df in S.items():
            df.to_excel(w, sheet_name=name[:31], index=False)
            w.sheets[name[:31]].freeze_panes = "A2"
            for col in w.sheets[name[:31]].columns:
                w.sheets[name[:31]].column_dimensions[col[0].column_letter].width = 18
    tr.drop(columns=["valid"]).to_csv(out / "decisions.csv", index=False, encoding="utf-8")
    if verbose:
        print(f"\nwrote {out / 'results.xlsx'} and decisions.csv\n\nAUDIT")
        for _, a in S["Audit"].iterrows():
            print(f"  [{'PASS' if a.passed else 'FAIL'}] {a.check}  {a.detail}")
        print("\nLEADERBOARD (main arm)")
        print(S["Leaderboard"].round(2).to_string(index=False))
        miss = PR[~PR.met] if len(PR) else PR
        if len(miss):
            print(f"\nPRECISION: {len(miss)} of {len(PR)} estimates miss their pre-set target. To meet all of them set "
                  f"REPS['main'] = {int(miss.main_reps_needed.max())} and run again (only the new repetitions are played).")
        else:
            print(f"\nPRECISION: all {len(PR)} key estimates meet their pre-set targets with main={REPS['main']} repetitions.")
    return S


# =============================================================================
# PROMPTS APPENDIX (reviewer 4.1: exact system and user prompts)
# =============================================================================

def write_prompts(path):
    o = ["# Appendix: exact prompts and settings (generated by experiment.py)\n",
         "Every API call contains exactly one system prompt and one user prompt. There is no conversation memory: "
         "the history the model sees is written into the user prompt. The opponent's identity is never revealed; "
         "in the reference arm the LLM receives the same prompt as in the main arm.\n",
         "## Models and settings\n", "| model | OpenRouter slug | provider (pinned) | temperature | hidden reasoning | max_tokens |",
         "|---|---|---|---|---|---|"]
    o += [f"| {k} | `{s}` | {r} | {TEMPERATURE} | {'disabled' if rs == 'off' else 'none (non-reasoning model)'} | "
          f"{MAX_TOKENS} (decision-only arm: {DECISION_MAX_TOKENS}) |" for k, (s, r, rs) in MODELS.items()]
    o += ["", "`max_tokens` limits visible output and any reasoning together; it is a ceiling, not a target.\n",
          "## Payoff matrices (canonical actions; first number = row player)\n",
          "| phase | A,A | B,A | A,B | B,B | dominant action | pure Nash equilibria | game type |", "|---|---|---|---|---|---|---|---|"]
    for p in PAYOFFS:
        c = classify(PAYOFFS[p])
        o.append(f"| {p} | " + " | ".join("%+d, %+d" % PAYOFFS[p][k] for k in (("A", "A"), ("B", "A"), ("A", "B"), ("B", "B")))
                 + f" | {c['dominant_action']} | {c['pure_nash_equilibria']} | {c['game_type']} |")
    o += ["", f"Each match has {NUM_ROUNDS} rounds in four phases of {NUM_ROUNDS // 4}. Phase orders: "
              + "; ".join(f"{k}: {v}" for k, v in ORDERS.items()) + ".\n",
          "## System prompt (all arms)\n", "```", SYSTEM_PROMPT.rstrip(), "```\n"]
    titles = {"main": "Main arm (reasoning, normal labels; also used in the reference arm)",
              "label": "Label-swap arm (the text 'Option A' denotes canonical action B and vice versa)",
              "decision": "Decision-only arm"}
    for arm, title in titles.items():
        o.append(f"## User prompt: {title}\n")
        for r, hist in ((1, []), (NUM_ROUNDS // 4 + 2, ["A", "B"]), (NUM_ROUNDS, ["B", "B", "A", "B", "A"])):
            ph = phase_of(r, "forward")
            o += [f"Example: round {r}, phase {ph}, forward order\n", "```", user_prompt(r, ph, hist, arm), "```\n"]
        if arm == "decision":
            o += ["The answer is additionally constrained by the API (OpenRouter `response_format`) to this JSON schema:\n",
                  "```json", json.dumps(DECISION_SCHEMA, indent=2), "```\n"]
    o += ["## Information available to the model\n",
          f"- Sees: the current matrix with both players' payoffs; round number and total number of rounds; the other "
          f"agent's last {HISTORY_WINDOW} choices ('no valid choice' if an answer was invalid).",
          "- Does not see: its own previous choices, cumulative scores, future matrices or when they change, "
          "the other agent's identity.\n", "## Parsing rule\n", PARSING_RULE, ""]
    Path(path).write_text("\n".join(o), encoding="utf-8")


# =============================================================================
# CHECK (real API, ~1 cent) and TEST (fake model, free)
# =============================================================================

async def check(client):
    out = HERE / "check"
    out.mkdir(exist_ok=True)
    for m in MODELS:
        for arm in ("main", "decision"):
            p = user_prompt(76, 4, ["A", "B", "A", "B", "B"], arm)
            try:
                mv = await ask(client, m, p, arm, dict(key="check", attempt="check", round=76, side="x", phase=4), out)
                rec = read_jsonl(out / "calls.jsonl")[-1]
                ok = mv is not None and not rec["reasoning_tokens"]
                print(f"[{'PASS' if ok else 'FAIL'}] {m:<9}{arm:<9} decision={mv} out_tokens={rec['completion_tokens']} "
                      f"hidden_reasoning={rec['reasoning_tokens']} cost=${rec['cost_usd'] or 0:.5f} "
                      f"served_by={rec['provider_returned']} version={rec['model_returned']}")
            except ApiFailure as e:
                print(f"[FAIL] {m:<9}{arm:<9} {e}")
        # temperature really applied? 3 identical prompts should give different texts
        p = user_prompt(76, 4, ["A", "B", "A", "B", "B"], "main")
        try:
            await asyncio.gather(*[ask(client, m, p, "main", dict(key="check", attempt="check", round=76, side="x",
                                                                   phase=4), out) for _ in range(3)])
            texts = {r["response"] for r in read_jsonl(out / "calls.jsonl")[-3:]}
            print(f"[{'PASS' if len(texts) > 1 else 'WARN'}] {m:<9}sampling: 3 identical prompts -> {len(texts)} different answers")
        except ApiFailure as e:
            print(f"[FAIL] {m:<9}sampling check: {e}")


class FakeError(Exception):
    def __init__(self, status):
        super().__init__(f"fake HTTP {status}")
        self.status_code = status


def fake_client(permanent_fail=0.0):
    """Plays a simple best-reply policy; injects invalid answers, truncations and API errors."""
    line = re.compile(r"You choose Option ([AB]), other agent chooses Option ([AB]): you get ([+-]\d+)")
    rnd = random.Random(1)

    async def create(**kw):
        await asyncio.sleep(0)
        user = kw["messages"][1]["content"]
        x = rnd.random()
        if x < permanent_fail:
            raise FakeError(400)
        if x < permanent_fail + 0.01:
            raise FakeError(503)                                        # transient: retried
        pay = {(a, b): int(p) for a, b, p in line.findall(user)}
        last = re.findall(r"Option ([AB])\]", user)
        move = max("AB", key=lambda a: pay[(a, last[0])] if last else min(pay[(a, "A")], pay[(a, "B")]))
        raw, finish = f"Analysis...\n**DECISION: OPTION {move}**", "stop"
        if "response_format" in kw:
            raw = json.dumps({"decision": f"OPTION {move}"})
        elif x > 0.98:
            raw = "Hard to say."                                         # invalid
        elif x > 0.97:
            raw, finish = "Analysis... I think Option", "length"         # truncated
        usage = SimpleNamespace(prompt_tokens=len(user) // 4, completion_tokens=len(raw) // 4,
                                completion_tokens_details=SimpleNamespace(reasoning_tokens=0), cost=1e-6)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw), finish_reason=finish)],
                               usage=usage, model=kw["model"],
                               model_extra={"provider": kw["extra_body"]["provider"]["order"][0]})

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test():
    global BACKOFF
    BACKOFF = 0
    out = HERE / "results_test"
    shutil.rmtree(out, ignore_errors=True)
    print("1) run with API failures: failed matches must NOT be saved")
    asyncio.run(run(fake_client(permanent_fail=0.0005), out, yes=True))
    rows1 = read_jsonl(out / "rounds.jsonl")
    n1 = len({r["key"] for r in rows1})
    # simulate a crash in the middle of writing a match: half of a match left on disk under a missing key
    missing = next(p["key"] for p in make_plan() if p["key"] not in {r["key"] for r in rows1})
    with open(out / "rounds.jsonl", "a", encoding="utf-8") as f:
        for r in rows1[:NUM_ROUNDS // 2]:
            f.write(json.dumps(dict(r, key=missing, attempt="halfwritten")) + "\n")
        f.write('{"key": "' + missing + '", "rou')                                  # cut mid-line
    print(f"\n2) run again: only the {len(make_plan()) - n1} missing matches are played")
    complete = asyncio.run(run(fake_client(), out, yes=True))
    S = analyze(out)
    R = complete_matches(read_jsonl(out / "rounds.jsonl"))
    pos_ok = all(sorted(o[i] for o in ORDERS.values()) == [1, 2, 3, 4] for i in range(4))
    pairs = Counter((o[i], o[i + 1]) for o in ORDERS.values() for i in range(3))
    williams_ok = pos_ok and len(pairs) == 12 and set(pairs.values()) == {1} and [1, 2, 3, 4] in ORDERS.values()
    num = re.compile(r"Option ([AB]), other agent chooses Option ([AB]): you get ([+-]\d+) points, other agent gets ([+-]\d+)")
    found = [(ph, sw, num.findall(matrix_text(ph, sw))) for ph in PAYOFFS for sw in (False, True)]
    matrix_ok = all(len(f) == 4 for _, _, f in found) and all(
        PAYOFFS[ph][(flip(a) if sw else a, flip(b) if sw else b)] == (int(x), int(y))
        for ph, sw, f in found for a, b, x, y in f)
    parser_ok = all(parse_decision(t) == e for t, e in [
        ("blah\nDECISION: OPTION A", "A"), ("**DECISION: OPTION B**", "B"), ("decision: option a ... DECISION: OPTION B", "B"),
        ('{"decision": "OPTION A"}', "A"), ("I would pick Option B", None), ("DECISION: OPTIONAL", None), ("", None)])
    write_prompts(out / "prompts_appendix.md")
    checks = [("the matrix shown equals the scoring table, normal and swapped labels", matrix_ok),
              ("parser: strict, last decision wins, no guessing", parser_ok),
              ("prompts appendix written", (out / "prompts_appendix.md").stat().st_size > 1000),
              ("phase orders form a Williams design that includes the original order", williams_ok),
              ("half-written match after a crash is ignored and replayed, no duplicate rounds",
               len(R) == NUM_ROUNDS * len(make_plan()) and missing in {r["key"] for r in R}),
              ("all result sheets present", all(s in S for s in ("Leaderboard", "Head-to-Head", "Phase Results",
               "Retaliation", "LLM vs Scripted", "Benchmark Gap", "Robustness", "Repetition Variability",
               "Strategic Indicators", "Precision", "Data Quality", "Game Classification", "Audit", "Protocol"))),
              ("all matches complete after the second run", complete),
              ("audit all PASS", bool(S["Audit"].passed.all())),
              ("invalid answers were voided, not defaulted", int(S["Void Rounds"].void_rounds.sum()) > 0),
              ("always_B earns 425 against always_A (reviewer's example)",
               int(S["Scripted vs Scripted"].query("order=='forward' and strategy=='always_B' and opponent=='always_A'")
                   .strategy_total.iloc[0]) == 425),
              ("phase 1 and 3 = prisoner's dilemma, 2 = not a dilemma, 4 = chicken",
               [classify(PAYOFFS[p])["game_type"][:5] for p in PAYOFFS] == ["Priso", "B dom", "Priso", "Anti-"])]
    print("\nTEST")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


# =============================================================================

def make_client():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("Set the OPENROUTER_API_KEY environment variable first.")
    from openai import AsyncOpenAI
    return AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, max_retries=0, timeout=TIMEOUT)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "test":
        sys.exit(0 if test() else 1)
    elif cmd == "check":
        asyncio.run(check(make_client()))
    elif cmd == "run":
        if asyncio.run(run(make_client(), OUT, yes="--yes" in sys.argv)):
            analyze(OUT)
    elif cmd == "analyze":
        analyze(OUT)
    elif cmd == "smoke":                      # real API, 4 matches, separate folder (~$0.5-1)
        if asyncio.run(run(make_client(), HERE / "results_smoke", yes=True, plan=smoke_plan())):
            analyze(HERE / "results_smoke", plan=smoke_plan())
    elif cmd == "prompts":
        write_prompts(HERE / "prompts_appendix.md")
        print(f"wrote {HERE / 'prompts_appendix.md'}")
    else:
        print(__doc__)
