# Equity Calculation — How It Works

This document explains exactly how `poker_equity.py` calculates equity,
from cards to percentages, step by step.

---

## What is Equity?

Equity is the probability that a player wins the pot if the hand is played
to the river with no more betting. It answers the question:

> "If we run out all remaining cards a million times, what fraction of those
> does each player win?"

Example: Player 1 has Ah Kh (ace-king suited). Player 2 has 7d 7c (a pair
of sevens). Before the flop, Player 1 has roughly 46% equity and Player 2
has roughly 54%. The pair is actually favoured, which surprises most people.

---

## Step 1 — Representing Cards as Numbers

Every card is stored as a single integer from 0 to 51.

```
card number = rank_index × 4 + suit_index
```

Ranks are indexed 0–12:  2=0, 3=1, 4=2, 5=3, 6=4, 7=5, 8=6, 9=7, T=8, J=9, Q=10, K=11, A=12
Suits are indexed 0–3:   c=0, d=1, h=2, s=3

Examples:
  2c = 0×4 + 0 = 0
  2d = 0×4 + 1 = 1
  Ah = 12×4 + 2 = 50
  As = 12×4 + 3 = 51

To get the rank back from a card number:   card // 4
To get the suit back from a card number:   card % 4

This encoding makes arithmetic fast and deck construction trivial:
the full deck is just the list [0, 1, 2, ..., 51].

---

## Step 2 — Evaluating a 5-Card Hand

`eval5(cards)` takes exactly 5 card numbers and returns a tuple that can
be compared directly. A higher tuple means a better hand.

### How the tuple is built

First, extract ranks and suits:
```
ranks = sorted list of (card // 4) for each card, descending
suits = list of (card % 4) for each card
```

Then count how many of each rank appear using a Counter:
```
groups = sorted by (count descending, rank descending)
sr = ranks in that order    e.g. for full house KKK88: [11, 7]
sc = their counts           e.g. [3, 2]
```

Check for flush: all 5 suits are the same → fl = True
Check for straight: 5 unique ranks spanning exactly 4 (max - min == 4)
  Special case: A-2-3-4-5 (the "wheel") — ace counts as low here

The return tuple starts with a category number 0–8:

| Category | Code | Tuple format        | Example           |
|----------|------|---------------------|-------------------|
| High card        | 0 | (0, r1,r2,r3,r4,r5) | 7 high            |
| One pair         | 1 | (1, pr,k1,k2,k3)   | Pair of 9s        |
| Two pair         | 2 | (2, p1,p2,k)        | Kings and Tens    |
| Three of a kind  | 3 | (3, tr,k1,k2)       | Trip Aces         |
| Straight         | 4 | (4, high_rank)      | 9-high straight   |
| Flush            | 5 | (5, r1,r2,r3,r4,r5) | Ace-high flush    |
| Full house       | 6 | (6, trips,pair)     | Aces full of Ks   |
| Four of a kind   | 7 | (7, quad,kicker)    | Quad Twos         |
| Straight flush   | 8 | (8, high_rank)      | Royal flush = (8,12) |

Because Python compares tuples left to right, (8, 12) > (7, 12) > (6, 12, 11)
automatically, with no special logic needed. The kicker cards in the tuple
break ties between same-category hands.

### Example: Two pair — Kings and Tens with a 5 kicker

Cards: Kh Kd Tc Ts 5h
ranks = [11, 11, 8, 8, 3]  (K=11, T=8, 5=3)
groups sorted: [(11,2), (8,2), (3,1)]
sc = [2, 2, 1]  → two pair detected
sr = [11, 8, 3]
return (2, 11, 8, 3)

Two pair Kings and Tens beats two pair Kings and Nines because
(2, 11, 8, 3) > (2, 11, 7, 3) at the second element.

---

## Step 3 — Best Hand from 7 Cards

Players have 2 hole cards. The board has up to 5 community cards.
That's up to 7 cards total but the hand is scored on the best 5.

`best_score(hole, board)` tries every possible combination of 5 cards
from those 7 and returns the highest scoring tuple.

Number of combinations: C(7,5) = 21

So it calls eval5 exactly 21 times and takes the maximum.

```python
return max(eval5(combo) for combo in combinations(all_cards, 5))
```

---

## Step 4 — The Equity Engine

`calc_equity(hands, board)` is the core function. It works differently
depending on how many community cards have already been dealt.

### The core loop (same for all methods)

For every possible completed board, score all players' hands, find the winner,
and award them a point. Ties split the point equally.

```
wins = [0, 0, 0, 0, 0]   (one per player)
total = 0

for each possible_board:
    scores = [best_score(hand, possible_board) for each player]
    winner = player with the highest score
    if tie: each tied player gets 1/N of the point
    wins[winner] += 1
    total += 1

equity[player] = wins[player] / total
```

### Method A — River (0 cards needed)

The board is complete. There is exactly one possible outcome.
Run the loop once. The result is 100% for the winner, 0% for everyone else.
(Or a split if two players tie.)

### Method B — Turn (1 card needed, ~44 possibilities)

One card is still to come. The remaining deck has 52 − 10 hole cards −
4 board cards = 38 cards. Loop through all 38 and run the core loop
for each. Total: 38 iterations. This is exhaustive — it checks every
single possible river card. The result is exact.

### Method C — Flop (2 cards needed, ~741 combinations)

Two cards still to come (turn + river). The remaining deck has
52 − 10 − 3 = 39 cards. We need all 2-card combinations from those:
C(39,2) = 741 combinations. Loop through all 741. Still exhaustive and exact.

### Method D — Preflop (5 cards needed, ~2.6 million combinations)

Before the flop, 5 community cards still need to come. The remaining deck has
52 − 10 = 42 cards. The number of 5-card combinations is C(42,5) = 850,668.
For 5 players that's 850,668 × 5 hand evaluations — too slow to do exhaustively.

Instead: **Monte Carlo simulation**. Randomly sample 20,000 boards, run the
core loop on each, and use the win rate as an estimate of the true probability.

With 20,000 samples the margin of error is roughly ±0.7% for any single
player's equity. Good enough for practical use.

```python
for _ in range(20_000):
    random_board = random.sample(remaining_deck, 5)
    _run(random_board)
```

### Why not Monte Carlo for everything?

Monte Carlo is an approximation. Exhaustive is exact. Once the flop is out
the number of remaining combinations is small enough (≤741) that exhaustive
is nearly instant, so we use it for accuracy.

---

## Step 5 — Delta Tracking (the arrows ▲ ▼)

After each street, `print_equity_table` compares the new equity to the
previous equity and prints an arrow showing who gained and who lost.

```
diff = new_equity% - old_equity%
▲ if diff > 0    (this street helped the player)
▼ if diff < 0    (this street hurt the player)
shown only if |diff| >= 0.05%
```

This is purely cosmetic — it doesn't affect the calculation.

---

## Full Example Walk-Through

Setup: 2 players.  Player 1: Ah Kh.  Player 2: 7d 7c.

**Preflop**
Deck has 48 remaining cards. Need 5 for the board.
Monte Carlo: 20,000 random boards sampled.
In roughly 46% of them, Ace-King makes a better hand (pair of aces, pair of
kings, straight, flush, etc.) than the sevens. Result: P1=46%, P2=54%.

**Flop: 7h 2d 9c**
Player 2 hit three-of-a-kind sevens. 
Deck now has 45 remaining cards, need 2. C(45,2) = 990 combinations.
Exhaustive loop: almost none of them let Ah Kh beat trip sevens.
Result: P1=5%, P2=95%.  P1 drops ▼41%.

**Turn: Jh**
Now Player 1 has A K J 9 — needs a T for a straight, or Q for Broadway.
Deck has 44 remaining, need 1. Loop through all 44 cards.
A Ten or Queen (8 outs out of 44) saves P1. 8/44 ≈ 18%.
Result: P1=18%, P2=82%.  P1 gains ▲13% (the straight draw is live).

**River: Th**
Player 1 makes: Ah Kh Jh Th — and checks the flush!
Actually A K Q J T of hearts is a Royal Flush. Wait — the Q wasn't on the board.
Let's say river is Qh instead.
Player 1: Ah Kh Qh Jh Th = Royal Flush (straight flush, ace-high).
eval5 returns (8, 12).  Player 2: best hand is still trip sevens = (3, ...).
(8, 12) > (3, ...) → Player 1 wins. P1=100%, P2=0%.

---

## Summary of the Key Design Choices

| Choice | Why |
|--------|-----|
| Cards as integers 0–51 | Fast arithmetic, no string parsing during simulation |
| eval5 returns a comparable tuple | No lookup tables, no if-chains for comparison — Python's built-in tuple comparison handles all tiebreakers automatically |
| best_score tries all C(7,5)=21 combos | Simpler and fast enough; no need for optimized 7-card evaluators at this scale |
| Exhaustive for turn/river | Exact results when computation is cheap |
| Monte Carlo for preflop | C(42,5) = 850k combos × 5 players is too slow; 20k samples gives <1% error |
| Split pots share the point fractionally | Correctly handles chop pots — the equity of a 3-way tie is exactly 1/3 each |
