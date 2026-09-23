"""Bracket-macro repair engine based on interval dynamic programming.

Given a sequence of bracket tokens (some positions locked), find the
well-formed nested sequence that

1. respects every locked position,
2. minimises the number of edited (unlocked) positions, and
3. among all minimum-cost results is lexicographically smallest under
   the explicit character order  ( < ) < [ < ] < { < } .

The solver never enumerates candidate replacements.  It uses the classic
context-free decomposition of balanced strings

    S  ->  open_t  S  close_t  S        (t in {(), [], {}})

over intervals [l, r):  dp[l][r] holds the best (cost, key, text) for the
substring s[l:r].  Because every candidate for a fixed interval has the
same length, per-interval (cost, key) minima compose into the global
optimum, giving an O(n^3) algorithm for n <= 160.
"""

from __future__ import annotations

OPENERS = "([{"
CLOSERS = ")]}"
MATCH_OPEN_TO_CLOSE = dict(zip(OPENERS, CLOSERS))
MATCH_CLOSE_TO_OPEN = {v: k for k, v in MATCH_OPEN_TO_CLOSE.items()}

# Explicit tie-break order required by the specification.
ALPHABET = "()[]{}"
RANK = {c: i for i, c in enumerate(ALPHABET)}

# Rank encoded as a single order-preserving character, so plain string
# comparison on keys reproduces the required lexicographic order exactly.
_KEY_CHR = "012345"
KEY_OF = {c: _KEY_CHR[RANK[c]] for c in ALPHABET}

INF = float("inf")


def solve(chars: list[str], locked: list[bool]) -> tuple[int, str] | None:
    """Return ``(edit_cost, repaired_text)`` or ``None`` if no repair exists.

    Only positions with ``locked[i] == False`` may be rewritten.  Locked
    positions are treated as hard constraints (infinite cost to change),
    never silently relaxed.
    """
    n = len(chars)

    # change[i][j]: cost of writing ALPHABET[j] at position i.
    change: list[list[float]] = []
    for i, c in enumerate(chars):
        if locked[i]:
            row = [INF] * 6
            row[RANK[c]] = 0.0
        else:
            row = [0.0 if d == c else 1.0 for d in ALPHABET]
        change.append(row)

    # dp tables over intervals [l, r); only even lengths are feasible.
    cost = [[INF] * (n + 1) for _ in range(n + 1)]
    key = [[""] * (n + 1) for _ in range(n + 1)]
    txt = [[""] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][i] = 0.0

    for length in range(2, n + 1, 2):
        for l in range(0, n - length + 1):
            r = l + length
            best_cost = INF
            best_key = ""
            best_txt = ""
            row_l = change[l]
            # Position l pairs with some k at odd distance, splitting the
            # interval into inner (l+1, k) and rest (k+1, r).
            for k in range(l + 1, r, 2):
                inner_cost = cost[l + 1][k]
                if inner_cost == INF:
                    continue
                rest_cost = cost[k + 1][r]
                if rest_cost == INF:
                    continue
                base = inner_cost + rest_cost
                inner_key = key[l + 1][k]
                inner_txt = txt[l + 1][k]
                rest_key = key[k + 1][r]
                rest_txt = txt[k + 1][r]
                row_k = change[k]
                for t in range(3):
                    cand_cost = row_l[2 * t] + row_k[2 * t + 1] + base
                    if cand_cost >= INF or cand_cost > best_cost:
                        continue
                    cand_key = (
                        _KEY_CHR[2 * t]
                        + inner_key
                        + _KEY_CHR[2 * t + 1]
                        + rest_key
                    )
                    if cand_cost < best_cost or cand_key < best_key:
                        best_cost = cand_cost
                        best_key = cand_key
                        best_txt = OPENERS[t] + inner_txt + CLOSERS[t] + rest_txt
            cost[l][r] = best_cost
            key[l][r] = best_key
            txt[l][r] = best_txt

    if cost[0][n] == INF:
        return None
    return int(cost[0][n]), txt[0][n]


def is_balanced(text: str) -> bool:
    """True iff ``text`` is a well-formed nested bracket sequence."""
    stack: list[str] = []
    for ch in text:
        if ch in MATCH_OPEN_TO_CLOSE:
            stack.append(ch)
        elif not stack or stack.pop() != MATCH_CLOSE_TO_OPEN[ch]:
            return False
    return not stack


def pairing(text: str) -> list[tuple[int, int]]:
    """Matched index pairs of a balanced string, sorted by opening index."""
    stack: list[int] = []
    pairs: list[tuple[int, int]] = []
    for i, ch in enumerate(text):
        if ch in MATCH_OPEN_TO_CLOSE:
            stack.append(i)
        else:
            pairs.append((stack.pop(), i))
    pairs.sort()
    return pairs
