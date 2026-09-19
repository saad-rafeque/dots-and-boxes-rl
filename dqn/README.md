# DQN

Dueling Double DQN with n-step returns, trained by self-play. No RL library — the loop, the
replay, the network and the board are all here.

| File | What it is |
|---|---|
| `game.py` | Board geometry, the 8 symmetries, batched move application, the fixed benchmark bot, and the 20-plane encoder. numpy throughout, numba for the hot loops. |
| `model.py` | The dueling network: residual conv trunk with global pooling, then a value head and an advantage head. 480,546 parameters. |
| `buffer.py` | Replay memory and the n-step window that assembles transitions and works out the sign. |
| `train_dqn.py` | Collect, learn, evaluate, checkpoint, plus the CLI. |
| `selftest.py` | Five checks that run before training starts. |
| `launcher.py` | Runs two configurations, one per GPU, for the Kaggle notebook. |

## Run it

```bash
python selftest.py       # ~20 s, must pass

python train_dqn.py --board 5 --out runs/dqn_a --hours 8.5 \
    --games 384 --channels 64 --blocks 6 --batch 512 \
    --lr 3e-4 --nstep 3 --eps-decay-steps 4000000 \
    --target-sync 2000 --updates 8 --eval-minutes 20

python train_dqn.py --eval-only --resume-from runs/dqn_a
```

Checkpoints land every 15 minutes and carry the network, the target network, the optimiser
state, the counters and the replay buffer, so a killed run resumes where it stopped. It also
saves on SIGTERM or on finding a file named `STOP` in the output folder — that's what let it
survive a full night on Kaggle with the browser closed.

## The loop

One iteration: play 8 moves across 384 boards in lockstep (one forward pass decides all 384),
push them through the n-step windows into replay, then take 8 gradient steps of 512 sampled
transitions. Every 2,000 updates the target network is refreshed. Every 20 minutes it plays 200
games against the benchmark bot and records the score. Every 15 minutes it checkpoints.

## Two things specific to this game

**The turn doesn't alternate.** Completing a box means you move again. Most n-step
implementations assume strict alternation and flip signs on a fixed schedule; this one walks the
trajectory and flips only where a move failed to capture. Each transition stores its own sign.
`selftest.py` walks a four-move sequence by hand to check it, because getting this wrong is
silent — the loss still falls while the agent learns backwards.

**The reward is paid per move.** Each move earns (boxes captured) / 25 straight away instead of
waiting 60 moves for the final margin. With the turn-change signs applied those payments sum to
exactly the final margin / 25, so it moves the signal earlier without distorting it. There's a
test asserting that identity on random games before every run.

## Settings, and why

| Setting | Run A | Run B | Reason |
|---|---|---|---|
| learning rate | 3e-4 | 1e-3 | A is the safe AdamW default; B tests whether bigger steps help or destabilise |
| n-step | 3 | 5 | Longer windows shorten the credit chain but add variance |
| epsilon | 1.0 → 0.05 | 1.0 → 0.05 | Never stops exploring entirely |
| epsilon decay | 4.0M steps | 1.5M steps | Both too short in practice — see the note in the top-level README |
| target sync | 2,000 | 1,000 | Less often is more stable but slower |
| updates per burst | 8 | 12 | Replay ratio; B trains harder per unit of experience |
| discount γ | 1.0 | 1.0 | Episodes are exactly 60 moves and a box is worth the same whenever it's won |
| batch | 512 | 512 | Fills a T4 without thrashing |
| replay | 600,000 | 600,000 | ~10,000 games of history |
| parallel games | 384 | 384 | One forward pass per move for all of them; this is what makes the GPU worth using |
| network | 64ch × 6 | 64ch × 6 | Identical, so the comparison is about the settings and not capacity |
| Huber β | 0.1 | 0.1 | Q targets contain outliers; squared error lets one sample dominate a batch |
| grad clip | 5.0 | 5.0 | Safety against a bad batch |

Run A won the head-to-head 200–0.
