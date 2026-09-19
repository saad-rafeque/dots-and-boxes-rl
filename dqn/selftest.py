"""Fast correctness checks for the DQN game core. Runs before any training."""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import game as G
from game import Geometry, Batch, Encoder, sides_from_drawn, C_IN
from model import DuelingQNet, q_from
from buffer import Replay, NStepTracker

ok = lambda m: print("PASS  " + m, flush=True)

for N in (3, 5, 6):
    g = Geometry(N)
    b = Batch(g, 16, np.random.default_rng(N))
    rng = np.random.default_rng(N)
    while not b.done.all():
        a = np.array([rng.choice(np.flatnonzero(b.legal()[i])) if not b.done[i] else 0
                      for i in range(16)])
        b.step(a)
    assert (b.score.sum(1) == g.B).all(), "every box scored exactly once"
    assert (b.drawn == 1).all(), "every line drawn"
ok("random games terminate with every box scored exactly once (3x3, 5x5, 6x6)")

g = Geometry(5)
b = Batch(g, 8, np.random.default_rng(0))
rng = np.random.default_rng(0)
for _ in range(30):
    b.step(np.array([rng.choice(np.flatnonzero(b.legal()[i])) for i in range(8)]))
X1 = b.encode()
enc = Encoder(g, 8)
X2 = enc(b.drawn, sides_from_drawn(g, b.drawn), b.owner, b.score, b.mover)
assert np.array_equal(X1, X2) and np.isfinite(X1).all()
ok("self-play encoder and replay encoder produce identical features")

# the sum of signed per-move rewards must equal the true final margin
b = Batch(g, 32, np.random.default_rng(3))
rng = np.random.default_rng(3)
acc = np.zeros(32)
sign = np.ones(32)
start = b.mover.copy()
while not b.done.all():
    live = ~b.done
    a = np.array([rng.choice(np.flatnonzero(b.legal()[i])) if live[i] else 0 for i in range(32)])
    r, same = b.step(a)
    acc += np.where(live, sign * r, 0.0)
    sign = np.where(live & ~same, -sign, sign)
true = b.margin() / g.B
assert np.allclose(acc, np.where(start == 1, true, -true), atol=1e-6), (acc[:4], true[:4])
ok("signed rewards telescope exactly to the final margin (no shaping bias)")

net = DuelingQNet(16, 2)
lf = torch.as_tensor((g.line_cell[:, 0] * g.S + g.line_cell[:, 1]).astype(np.int64))
x = torch.randn(4, C_IN, g.S, g.S)
adv, v = net(x, lf)
legal = torch.zeros(4, g.L, dtype=torch.bool)
legal[:, :7] = True
q = q_from(adv, v, legal)
assert torch.isfinite(q[:, :7]).all() and (q[:, 7:] < -1e8).all()
ok("dueling head: legal Q finite, illegal moves masked out")

tr = NStepTracker(1, 3, 1.0)
st = lambda i: (np.zeros(g.L, np.uint8) + i, np.zeros(g.B, np.int8), np.zeros(2, np.int16), np.int8(1))
for i, (r, same) in enumerate([(0.0, False), (0.2, True), (0.0, False), (0.0, False)]):
    tr.push(0, st(i), i, r, same)
out = tr.pop_ready(0, st(9), True)
assert len(out) == 4 and out[0]["k"] == 3
# walk it by hand from step 0. move 0 does not capture -> turn passes to the
# opponent; move 1 captures 0.2 for the OPPONENT and keeps their turn; move 2
# does not capture -> the turn comes back.  So from step 0's point of view the
# accumulated reward is -0.2 and the state we bootstrap from is ours again.
assert abs(out[0]["r_acc"] + 0.2) < 1e-6, out[0]["r_acc"]
assert out[0]["sign"] == +1, out[0]["sign"]
assert out[0]["n_drawn"][0] == 3 and out[0]["done"] == 0, "bootstrap from the 3-steps-later state"
# the last transition has nothing after it, so it must bootstrap from the end
assert out[-1]["done"] == 1 and out[-1]["k"] == 1
ok("n-step returns carry the correct sign through the extra-turn rule")
print("ALL SELF-TESTS PASSED", flush=True)
