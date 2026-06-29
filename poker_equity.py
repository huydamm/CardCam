#!/usr/bin/env python3
"""
Poker Hand Equity Calculator — 5-Player Texas Hold'em
Tracks equity through preflop, flop, turn, and river with Monte Carlo simulation.
"""

import random
import sys
from itertools import combinations
from collections import Counter

RANKS = '23456789TJQKA'
SUITS = 'cdhs'
HAND_NAMES = [
    'High Card', 'One Pair', 'Two Pair', 'Three of a Kind',
    'Straight', 'Flush', 'Full House', 'Four of a Kind', 'Straight Flush',
]
N_SIM = 20_000   # Monte Carlo sims for preflop; flop/turn/river use exhaustive


# ── Card helpers ──────────────────────────────────────────────────────────────

def card_str(c: int) -> str:
    return RANKS[c // 4] + SUITS[c % 4].upper()


def parse_card(s: str) -> int:
    s = s.strip()
    if len(s) != 2:
        raise ValueError(f"'{s}': use Rank+Suit, e.g. Ah Kd Tc 2s")
    r, su = s[0].upper(), s[1].lower()
    if r not in RANKS:
        raise ValueError(f"Unknown rank '{r}' — valid: 2 3 4 5 6 7 8 9 T J Q K A")
    if su not in SUITS:
        raise ValueError(f"Unknown suit '{su}' — valid: c d h s")
    return RANKS.index(r) * 4 + SUITS.index(su)


# ── Hand evaluator ────────────────────────────────────────────────────────────

def eval5(cards) -> tuple:
    """Score exactly 5 cards; higher tuple = better hand."""
    ranks = sorted([c // 4 for c in cards], reverse=True)
    suits = [c % 4 for c in cards]
    cnt = Counter(ranks)
    grps = sorted(cnt.items(), key=lambda x: (x[1], x[0]), reverse=True)
    sr = [r for r, _ in grps]   # ranks sorted by (count desc, rank desc)
    sc = [c for _, c in grps]   # corresponding counts
    fl = len(set(suits)) == 1
    uniq = sorted(set(ranks), reverse=True)
    st, sh = False, None
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            st, sh = True, uniq[0]
        elif set(uniq) == {12, 3, 2, 1, 0}:    # wheel: A-2-3-4-5
            st, sh = True, 3
    if st and fl:                             return (8, sh)
    if sc[0] == 4:                            return (7,) + tuple(sr)
    if sc[0] == 3 and len(sc) > 1 and sc[1] == 2:
                                              return (6,) + tuple(sr)
    if fl:                                    return (5,) + tuple(ranks)
    if st:                                    return (4, sh)
    if sc[0] == 3:                            return (3,) + tuple(sr)
    if sc[0] == 2 and len(sc) > 1 and sc[1] == 2:
                                              return (2,) + tuple(sr)
    if sc[0] == 2:                            return (1,) + tuple(sr)
    return (0,) + tuple(sr)


def best_score(hole: list, board: list):
    """Best 5-card score from hole cards + board (needs >= 5 cards combined)."""
    all_c = hole + board
    if len(all_c) < 5:
        return None
    return max(eval5(combo) for combo in combinations(all_c, 5))


# ── Equity engine ─────────────────────────────────────────────────────────────

def calc_equity(hands: list, board: list) -> tuple:
    """
    Return (equities, cur_scores).
      equities  — list of floats 0-1 summing to 1
      cur_scores — best hand scores with current board (None if < 5 cards)
    Uses exhaustive enumeration for turn/river, Monte Carlo for preflop/flop.
    """
    n = len(hands)
    wins = [0.0] * n
    total = 0
    used = {c for h in hands for c in h} | set(board)
    deck = [c for c in range(52) if c not in used]
    needed = 5 - len(board)

    def _run(full_board: list):
        nonlocal total
        scores = [best_score(h, full_board) for h in hands]
        mx = max(scores)
        ws = [i for i, s in enumerate(scores) if s == mx]
        share = 1.0 / len(ws)
        for w in ws:
            wins[w] += share
        total += 1

    if needed == 0:
        _run(board)
    elif needed <= 2:           # exhaustive (<=741 combos on flop, <=38 on turn)
        for combo in combinations(deck, needed):
            _run(board + list(combo))
    else:                       # Monte Carlo for preflop (need 5) and flop (need 3, ~18k combos)
        random.seed()
        for _ in range(N_SIM):
            _run(board + random.sample(deck, needed))

    eq = [w / total for w in wins]
    cur = [best_score(h, board) for h in hands] if board else [None] * n
    return eq, cur


# ── Display ───────────────────────────────────────────────────────────────────

def equity_bar(pct: float, width: int = 22) -> str:
    filled = round(pct / 100 * width)
    return '█' * filled + '░' * (width - filled)


def print_equity_table(hands: list, board: list, stage: str, prev_eq=None) -> list:
    """Print an equity table and return the equity list (0-1) for delta tracking."""
    eq, scores = calc_equity(hands, board)

    board_part = ('  ' + ' '.join(card_str(c) for c in board)) if board else ''
    title = f'  {stage}{board_part}'
    sep = '=' * 70

    print()
    print(sep)
    print(title)
    print(sep)

    for i, (hand, e, score) in enumerate(zip(hands, eq, scores)):
        pct = e * 100
        hand_s = ' '.join(card_str(c) for c in hand)
        b = equity_bar(pct)
        hn = HAND_NAMES[score[0]] if score else ''

        if prev_eq is not None:
            diff = pct - prev_eq[i] * 100
            if abs(diff) >= 0.05:
                arrow = '▲' if diff > 0 else '▼'
                delta = f' {arrow}{abs(diff):4.1f}%'
            else:
                delta = '        '
        else:
            delta = ''

        print(f'  P{i + 1}  {hand_s}  {b}  {pct:5.1f}%{delta}  {hn}')

    print()
    return eq


# ── Input helpers ─────────────────────────────────────────────────────────────

def prompt_cards(prompt_text: str, n: int, used: set) -> list:
    while True:
        try:
            raw = input(prompt_text).strip()
        except (EOFError, KeyboardInterrupt):
            print('\n  Goodbye.')
            sys.exit(0)
        if not raw:
            continue
        parts = raw.split()
        if len(parts) != n:
            print(f'  Enter exactly {n} card(s) separated by spaces. Example: Ah Kd')
            continue
        try:
            cards = [parse_card(p) for p in parts]
        except ValueError as e:
            print(f'  Error: {e}')
            continue
        dupes = [c for c in cards if c in used]
        if dupes:
            print(f'  Already in play: {" ".join(card_str(c) for c in dupes)}')
            continue
        return cards


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print()
    print('=' * 56)
    print('  POKER EQUITY CALCULATOR  —  5-Player Texas Hold\'em')
    print('=' * 56)
    print('  Card format: [Rank][Suit]')
    print('  Ranks : 2 3 4 5 6 7 8 9 T J Q K A')
    print('  Suits : c=clubs  d=diamonds  h=hearts  s=spades')
    print('  Example: Ah Kd  =  Ace of hearts, King of diamonds')
    print()

    n_players = 5
    hands = []
    used: set = set()

    print('  Enter each player\'s 2 hole cards:')
    for i in range(n_players):
        cards = prompt_cards(f'    Player {i + 1}: ', 2, used)
        hands.append(cards)
        used.update(cards)

    print(f'\n  Calculating preflop equity ({N_SIM:,} simulations)...')
    prev = print_equity_table(hands, [], 'PREFLOP')

    # ── Flop ──────────────────────────────────────────────────────────────────
    input('  [ Press Enter to deal the flop ] ')
    flop = prompt_cards('  Flop (3 cards): ', 3, used)
    used.update(flop)
    prev = print_equity_table(hands, flop, 'FLOP', prev)

    # ── Turn ──────────────────────────────────────────────────────────────────
    input('  [ Press Enter to deal the turn ] ')
    turn = prompt_cards('  Turn (1 card): ', 1, used)
    used.update(turn)
    board = flop + turn
    prev = print_equity_table(hands, board, 'TURN', prev)

    # ── River ─────────────────────────────────────────────────────────────────
    input('  [ Press Enter to deal the river ] ')
    river = prompt_cards('  River (1 card): ', 1, used)
    board = flop + turn + river
    prev = print_equity_table(hands, board, 'RIVER', prev)

    # ── Winner announcement ───────────────────────────────────────────────────
    eq, scores = calc_equity(hands, board)
    winners = [i for i, e in enumerate(eq) if e > 0.99]
    split_pot = [i for i, e in enumerate(eq) if 0.005 < e < 0.995]

    print('  ' + '-' * 50)
    if winners:
        for w in winners:
            hs = ' '.join(card_str(c) for c in hands[w])
            sn = HAND_NAMES[scores[w][0]]
            print(f'  ★  WINNER: Player {w + 1}  [ {hs} ]  —  {sn}')
    elif split_pot:
        ps = ' and '.join(f'Player {i + 1}' for i in split_pot)
        print(f'  ★  SPLIT POT: {ps}')
    print()


if __name__ == '__main__':
    main()
