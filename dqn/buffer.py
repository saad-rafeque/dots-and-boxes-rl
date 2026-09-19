"""N-step transition store for two-player DQN.

The turn does not alternate reliably in Dots & Boxes -- completing a box gives
you another move -- so a transition cannot assume the next state belongs to the
opponent.  Every entry therefore carries `sign`: +1 if the bootstrap state is
the SAME player's turn as the state the transition started from, -1 if it is the
opponent's.  The n-step target is

    y = r_acc + sign * gamma^k * max_a Q(s_k, a)        (0 if the game ended)

where r_acc already has the per-step signs folded in.
"""
import numpy as np


class Replay:
    def __init__(self, geo, capacity):
        L, B = geo.L, geo.B
        self.cap = int(capacity)
        self.drawn = np.zeros((self.cap, L), np.uint8)
        self.owner = np.zeros((self.cap, B), np.int8)
        self.score = np.zeros((self.cap, 2), np.int16)
        self.mover = np.zeros(self.cap, np.int8)
        self.action = np.zeros(self.cap, np.int16)
        self.r_acc = np.zeros(self.cap, np.float32)
        self.sign = np.zeros(self.cap, np.int8)
        self.k = np.zeros(self.cap, np.int8)
        self.done = np.zeros(self.cap, np.uint8)
        self.n_drawn = np.zeros((self.cap, L), np.uint8)
        self.n_owner = np.zeros((self.cap, B), np.int8)
        self.n_score = np.zeros((self.cap, 2), np.int16)
        self.n_mover = np.zeros(self.cap, np.int8)
        self.n = self.ptr = self.total = 0

    FIELDS = ("drawn", "owner", "score", "mover", "action", "r_acc", "sign",
              "k", "done", "n_drawn", "n_owner", "n_score", "n_mover")

    def add_many(self, **kw):
        m = len(kw["action"])
        if m == 0:
            return
        idx = (self.ptr + np.arange(m)) % self.cap
        for f in self.FIELDS:
            getattr(self, f)[idx] = kw[f]
        self.ptr = int((self.ptr + m) % self.cap)
        self.n = min(self.cap, self.n + m)
        self.total += m

    def sample(self, batch, rng):
        i = rng.integers(0, self.n, size=batch)
        return {f: getattr(self, f)[i] for f in self.FIELDS}

    def save(self, path):
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, n=self.n, ptr=self.ptr, total=self.total,
                            **{f: getattr(self, f)[:self.n] for f in self.FIELDS})
        import os
        os.replace(tmp, path)

    def load(self, path):
        d = np.load(path)
        n = min(int(d["n"]), self.cap)
        for f in self.FIELDS:
            getattr(self, f)[:n] = d[f][-n:] if len(d[f]) > n else d[f][:n]
        self.n = n
        self.ptr = n % self.cap
        self.total = int(d["total"])


class NStepTracker:
    """Per-game sliding window that turns raw 1-step transitions into n-step ones."""

    def __init__(self, G, n, gamma=1.0):
        self.G, self.n, self.gamma = G, int(n), float(gamma)
        self.buf = [[] for _ in range(G)]

    def reset_game(self, g):
        self.buf[g] = []

    def push(self, g, state, action, reward, same_mover):
        """state = (drawn, owner, score, mover) copies for THIS decision."""
        self.buf[g].append([state, action, reward, same_mover])

    def pop_ready(self, g, next_state, terminal):
        """Emit every transition whose n-step window is complete.
        On a terminal step everything still buffered is emitted."""
        out = []
        b = self.buf[g]
        limit = len(b) if terminal else len(b) - self.n
        for t in range(max(0, limit)):
            st, a, _, _ = b[t]
            sign = 1
            r_acc = 0.0
            g_pow = 1.0
            steps = 0
            for i in range(t, len(b)):
                r_acc += sign * g_pow * b[i][2]
                g_pow *= self.gamma
                steps += 1
                if not b[i][3]:          # no capture -> the turn changed
                    sign = -sign
                if steps == self.n:
                    break
            j = t + steps
            if j < len(b):               # bootstrap from a state we still hold
                ns, done = b[j][0], 0
            else:                        # ran off the end of the trajectory
                ns, done = next_state, (1 if terminal else 0)
            out.append(dict(drawn=st[0], owner=st[1], score=st[2], mover=st[3],
                            action=a, r_acc=r_acc, sign=sign, k=steps, done=done,
                            n_drawn=ns[0], n_owner=ns[1], n_score=ns[2], n_mover=ns[3]))
        if terminal:
            self.buf[g] = []
        else:
            del b[:max(0, limit)]
        return out
