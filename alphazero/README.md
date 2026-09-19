# AlphaZero engine

Policy/value network guiding a PUCT tree search, with an exact solver taking over in the
endgame. One file, around 2,800 lines, numba-compiled hot paths.

```bash
python nightmare_dab.py test                 # correctness suite
python nightmare_dab.py test --quick         # the fast subset
python nightmare_dab.py bench --board 6      # speed benchmark
python nightmare_dab.py play --board 5 --model ../models/best_5x5.pt --think 8
python nightmare_dab.py train --board 5 --out runs/b5 --hours 9.5
```

## How it plays

The network gives move priors and a value estimate; PUCT search turns those into a much better
move by simulating a few hundred continuations. Where it can, it skips estimating entirely:

- board finished → exact utility
- position is only chains and loops → chain theory gives the exact value
- 14 or fewer lines left at a leaf → alpha-beta within a node budget
- 24 or fewer lines left at the root (40 when playing a human) → full exact solve

The solver is negamax alpha-beta with a transposition table keyed only on the drawn lines (the
value is score-independent, so the same key serves every score), dominance pruning, chain
de-duplication, and the standard loony-endgame results: an open chain of n ≤ 2 is worth
−(n+V′), a chain of n ≥ 3 is −(n−2+|V′+2|), and a loop of n ≥ 4 is −(n−4+|V′+4|).

Roughly 40% of self-play training targets come back proven rather than estimated, which is most
of why the endgame play is exact rather than merely good.

## Training

Self-play with batched MCTS across 512 games at once, one GPU batch per round, virtual loss, FPU
reduction, Dirichlet root noise and playout-cap randomisation. Endgame solving runs on worker
threads so the GPU never waits on it. Replay buffer with 8-fold symmetry augmentation applied at
sample time. A new network only replaces the reference opponent after winning a 64-game match at
≥ 0.55, which stops the loop chasing its own noise.

The checkpoints in `../models/` come from 19.5 hours on two T4s: 397,094 self-play games on 5×5
and 179,109 on 6×6.
