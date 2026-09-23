"""Solver tests: handcrafted cases plus exhaustive brute-force cross-checks.

The brute force enumerates every replacement of the unlocked positions and
picks the optimum under the same (cost, lexicographic) objective, so the
interval DP is validated against ground truth on all short inputs.
"""

from __future__ import annotations

import itertools
import random

import pytest

from app.repair import ALPHABET, RANK, is_balanced, pairing, solve


def brute_force(chars: list[str], locked: list[bool]) -> tuple[int, str] | None:
    """Reference optimum by full enumeration of unlocked assignments."""
    free = [i for i, lk in enumerate(locked) if not lk]
    best: tuple[int, tuple[int, ...], str] | None = None
    for assign in itertools.product(ALPHABET, repeat=len(free)):
        text = list(chars)
        for i, c in zip(free, assign):
            text[i] = c
        s = "".join(text)
        if not is_balanced(s):
            continue
        cost = sum(a != b for a, b in zip(s, chars))
        key = tuple(RANK[c] for c in s)
        if best is None or (cost, key) < (best[0], best[1]):
            best = (cost, key, s)
    if best is None:
        return None
    return best[0], best[2]


def assert_consistent(chars: list[str], locked: list[bool], result) -> None:
    """Every solver answer must be balanced, respect locks and report cost."""
    if result is None:
        return
    cost, text = result
    assert len(text) == len(chars)
    assert is_balanced(text)
    for i, lk in enumerate(locked):
        if lk:
            assert text[i] == chars[i], "locked position was modified"
    assert cost == sum(a != b for a, b in zip(text, chars))


# ---------------------------------------------------------------------------
# handcrafted cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "chars, locked, expected",
    [
        # already balanced -> untouched
        ("([])", None, (0, "([])")),
        ("()[]{}", None, (0, "()[]{}")),
        # single wrong closer: cheapest fix wins, not the lexicographically
        # smaller but more expensive rewrites
        ("(]", None, (1, "()")),
        ("[[", None, (1, "[]")),
        # cost-1 ties between () and {}: '(' < '{' decides
        ("{)", None, (1, "()")),
        # all three types cost 2 -> lexicographic minimum ()
        ("}{", None, (2, "()")),
        (")(", None, (2, "()")),
        # crossed closers: min cost 2, several candidates, (()) is smallest
        ("([)]", None, (2, "(())")),
        # tie between [][] and [[]] at cost 2: '[' < ']' decides
        ("]]]]", None, (2, "[[]]")),
        # locked positions are hard constraints
        ("([", [True, False], (1, "()")),
        ("([)]", [True, False, False, False], (2, "(())")),
        ("([)]", [False, True, False, False], (2, "([])")),
        # all locked and already valid
        ("([])", [True] * 4, (0, "([])")),
    ],
)
def test_handcrafted_ok(chars, locked, expected):
    locked = locked or [False] * len(chars)
    assert solve(list(chars), locked) == expected


@pytest.mark.parametrize(
    "chars, locked",
    [
        ("((", [True, True]),          # all locked, unbalanced
        ("([)]", [True] * 4),         # all locked, crossed
        (")(", [True, False]),         # locked closer at position 0
        ("()([", [False, False, False, True]),  # locked trailing opener
        ("(]", [True, True]),
    ],
)
def test_no_repair(chars, locked):
    assert solve(list(chars), locked) is None


def test_locked_never_relaxed_even_when_cheaper():
    chars = list("(]))")
    # Unlocked, the optimum is a single edit; locking position 1 forbids it,
    # so the solver must pay more instead of silently touching the lock.
    assert solve(chars, [False] * 4) == (1, "(())")
    result = solve(chars, [False, True, False, False])
    assert result == (2, "[]()")
    assert result[1][1] == "]"


# ---------------------------------------------------------------------------
# exhaustive cross-checks against brute force
# ---------------------------------------------------------------------------

def run_cross_check(chars, locked):
    expected = brute_force(list(chars), locked)
    got = solve(list(chars), locked)
    assert_consistent(list(chars), locked, got)
    assert got == expected


def test_exhaustive_n2_all_locks():
    for chars in itertools.product(ALPHABET, repeat=2):
        for mask in itertools.product([False, True], repeat=2):
            run_cross_check(list(chars), list(mask))


def test_exhaustive_n4_selected_locks():
    masks = [
        [False, False, False, False],
        [True, True, True, True],
        [True, False, True, False],
        [False, True, False, True],
        [True, True, False, False],
    ]
    for chars in itertools.product(ALPHABET, repeat=4):
        for mask in masks:
            run_cross_check(list(chars), mask)


def test_random_n6_cross_check():
    rng = random.Random(20260923)
    for _ in range(120):
        chars = [rng.choice(ALPHABET) for _ in range(6)]
        # keep at most 4 unlocked positions so brute force stays cheap
        locked = [rng.random() < 0.4 for _ in range(6)]
        while sum(not lk for lk in locked) > 4:
            locked[rng.randrange(6)] = True
        run_cross_check(chars, locked)


def test_random_n8_cross_check():
    rng = random.Random(20260924)
    for _ in range(50):
        chars = [rng.choice(ALPHABET) for _ in range(8)]
        locked = [rng.random() < 0.5 for _ in range(8)]
        while sum(not lk for lk in locked) > 4:
            locked[rng.randrange(8)] = True
        run_cross_check(chars, locked)


# ---------------------------------------------------------------------------
# invariants on larger inputs
# ---------------------------------------------------------------------------

def test_random_medium_invariants():
    rng = random.Random(7)
    for n in (10, 20, 40, 80):
        for _ in range(20):
            chars = [rng.choice(ALPHABET) for _ in range(n)]
            locked = [rng.random() < 0.3 for _ in range(n)]
            assert_consistent(chars, locked, solve(chars, locked))


def test_max_length_all_openers():
    n = 160
    chars = ["("] * n
    cost, text = solve(chars, [False] * n)
    assert cost == n // 2
    assert text == "(" * (n // 2) + ")" * (n // 2)


def test_max_length_already_balanced():
    chars = list("()" * 80)
    assert solve(chars, [False] * 160) == (0, "()" * 80)
    assert solve(chars, [True] * 160) == (0, "()" * 80)


def test_max_length_random_smoke():
    rng = random.Random(160)
    chars = [rng.choice(ALPHABET) for _ in range(160)]
    locked = [rng.random() < 0.5 for _ in range(160)]
    assert_consistent(chars, locked, solve(chars, locked))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_pairing():
    assert pairing("(())") == [(0, 3), (1, 2)]
    assert pairing("()[]{}") == [(0, 1), (2, 3), (4, 5)]
    assert pairing("([]){()}") == [(0, 3), (1, 2), (4, 7), (5, 6)]


def test_is_balanced():
    assert is_balanced("([]){()}")
    assert not is_balanced("([)]")
    assert not is_balanced("(")
    assert not is_balanced(")")
    assert is_balanced("")
