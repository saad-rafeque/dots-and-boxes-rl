"""Dots & Boxes as a vectorised RL environment.

Boards are counted in BOXES: N=5 -> 5x5 boxes, 60 lines, 25 boxes.
Everything is batched: one call steps G independent games at once, which is what
keeps a GPU busy during DQN self-play.

Convention used everywhere in this package
------------------------------------------
Rewards and values are always from the point of view of THE PLAYER TO MOVE in
the state they are attached to, expressed as a fraction of the total boxes.
The reward for a move is (boxes that move captured) / B, so the rewards of a
whole game, with the sign flipped every time the turn changes, sum exactly to
the final margin / B.  Completing a box gives you another turn; that is why the
turn does not simply alternate and why `same_mover` is returned by `step`.
"""
import numpy as np
from numba import njit

__all__ = ["Geometry", "Batch", "C_IN", "heuristic_moves"]

C_IN = 20   # see Batch.encode for the plane list


class Geometry:
    """Line/box index maps for an N x N board, plus the 8 board symmetries."""

    def __init__(self, N):
        assert 2 <= N <= 7
        self.N = N
        self.B = B = N * N
        self.L = L = 2 * N * (N + 1)
        self.S = S = 2 * N + 1
        self.NH = N * (N + 1)

        line_cell = np.zeros((L, 2), np.int32)
        for r in range(N + 1):
            for c in range(N):
                line_cell[self.h(r, c)] = (2 * r, 2 * c + 1)
        for r in range(N):
            for c in range(N + 1):
                line_cell[self.v(r, c)] = (2 * r + 1, 2 * c)

        box_cell = np.zeros((B, 2), np.int32)
        box_lines = np.zeros((B, 4), np.int32)
        for r in range(N):
            for c in range(N):
                b = r * N + c
                box_cell[b] = (2 * r + 1, 2 * c + 1)
                box_lines[b] = (self.h(r, c), self.h(r + 1, c),
                                self.v(r, c), self.v(r, c + 1))

        line_box = np.full((L, 2), -1, np.int32)
        for b in range(B):
            for l in box_lines[b]:
                if line_box[l, 0] < 0:
                    line_box[l, 0] = b
                else:
                    line_box[l, 1] = b

        cell_line = np.full((S, S), -1, np.int32)
        cell_box = np.full((S, S), -1, np.int32)
        for l in range(L):
            cell_line[line_cell[l, 0], line_cell[l, 1]] = l
        for b in range(B):
            cell_box[box_cell[b, 0], box_cell[b, 1]] = b

        line_perm = np.zeros((8, L), np.int32)
        box_perm = np.zeros((8, B), np.int32)
        for t in range(8):
            for l in range(L):
                r, c = self._tf(t, line_cell[l, 0], line_cell[l, 1])
                line_perm[t, l] = cell_line[r, c]
            for b in range(B):
                r, c = self._tf(t, box_cell[b, 0], box_cell[b, 1])
                box_perm[t, b] = cell_box[r, c]
        assert (line_perm >= 0).all() and (box_perm >= 0).all()

        self.line_cell, self.box_cell = line_cell, box_cell
        self.box_lines, self.line_box = box_lines, line_box
        self.line_perm, self.box_perm = line_perm, box_perm
        self.line_rc = (line_cell[:, 0], line_cell[:, 1])
        self.box_rc = (box_cell[:, 0], box_cell[:, 1])

    def h(self, r, c):
        return r * self.N + c

    def v(self, r, c):
        return self.NH + r * (self.N + 1) + c

    def _tf(self, t, r, c):
        m = self.S - 1
        return [(r, c), (c, m - r), (m - r, m - c), (m - c, r),
                (r, m - c), (m - r, c), (c, r), (m - c, m - r)][t]

    def describe(self, l):
        if l < self.NH:
            return "h %d %d" % divmod(l, self.N)
        r, c = divmod(l - self.NH, self.N + 1)
        return "v %d %d" % (r, c)


# ---------------------------------------------------------------- numba core

@njit(cache=True)
def _apply(drawn, sides, owner, score, mover, done, actions, line_box, B):
    """Apply one action per game. Returns (captures, same_mover)."""
    G = drawn.shape[0]
    caps = np.zeros(G, np.int64)
    same = np.zeros(G, np.uint8)
    for g in range(G):
        if done[g]:
            continue
        m = actions[g]
        drawn[g, m] = 1
        k = 0
        for j in range(2):
            b = line_box[m, j]
            if b >= 0:
                sides[g, b] += 1
                if sides[g, b] == 4:
                    owner[g, b] = mover[g]
                    k += 1
        caps[g] = k
        if k > 0:
            score[g, mover[g] - 1] += k
            same[g] = 1
        else:
            mover[g] = 3 - mover[g]
        nleft = 0
        for b in range(B):
            if sides[g, b] < 4:
                nleft = 1
                break
        if nleft == 0:
            done[g] = True
    return caps, same


@njit(cache=True)
def _components(drawn, sides, line_box, box_lines, B, comp, size, formed, ground, stack):
    """Connected groups of open boxes joined by undrawn lines, per game.

    comp[g,b]   component id of box b (-1 if the box is finished)
    size[g,c]   number of boxes in component c
    formed[g,c] 1 if every box in it already has exactly 2 sides (a real chain/loop)
    ground[g,c] number of undrawn lines leaving the component to the board edge
    Returns the component count per game.
    """
    G = drawn.shape[0]
    ncomp = np.zeros(G, np.int64)
    for g in range(G):
        for b in range(B):
            comp[g, b] = -1
        nc = 0
        for s in range(B):
            if sides[g, s] >= 4 or comp[g, s] >= 0:
                continue
            comp[g, s] = nc
            size[g, nc] = 0
            formed[g, nc] = 1
            ground[g, nc] = 0
            top = 0
            stack[g, top] = s
            top += 1
            while top > 0:
                top -= 1
                b = stack[g, top]
                size[g, nc] += 1
                if sides[g, b] != 2:
                    formed[g, nc] = 0
                for j in range(4):
                    l = box_lines[b, j]
                    if drawn[g, l] != 0:
                        continue
                    o = line_box[l, 0]
                    if o == b:
                        o = line_box[l, 1]
                    if o < 0:
                        ground[g, nc] += 1
                    elif comp[g, o] < 0:
                        comp[g, o] = nc
                        stack[g, top] = o
                        top += 1
            nc += 1
        ncomp[g] = nc
    return ncomp


@njit(cache=True)
def _heuristic(drawn, sides, line_box, box_lines, L, B, done, out):
    """A decent casual opponent: take free boxes, then play safe, else give away
    as few boxes as possible.  Fixed forever so eval numbers stay comparable."""
    G = drawn.shape[0]
    for g in range(G):
        if done[g]:
            out[g] = -1
            continue
        pick = -1
        # a capture that does not open another box
        for b in range(B):
            if sides[g, b] == 3:
                l = -1
                for j in range(4):
                    if drawn[g, box_lines[b, j]] == 0:
                        l = box_lines[b, j]
                        break
                o = line_box[l, 0]
                if o == b:
                    o = line_box[l, 1]
                if o < 0 or sides[g, o] != 2:
                    pick = l
                    break
        if pick < 0:
            for b in range(B):
                if sides[g, b] == 3:
                    for j in range(4):
                        if drawn[g, box_lines[b, j]] == 0:
                            pick = box_lines[b, j]
                            break
                    break
        if pick < 0:                       # a safe line
            n = 0
            for l in range(L):
                if drawn[g, l] == 0:
                    ok = True
                    for j in range(2):
                        b = line_box[l, j]
                        if b >= 0 and sides[g, b] >= 2:
                            ok = False
                    if ok:
                        n += 1
            if n > 0:
                pickn = np.random.randint(n)
                n = 0
                for l in range(L):
                    if drawn[g, l] == 0:
                        ok = True
                        for j in range(2):
                            b = line_box[l, j]
                            if b >= 0 and sides[g, b] >= 2:
                                ok = False
                        if ok:
                            if n == pickn:
                                pick = l
                                break
                            n += 1
        if pick < 0:                       # open the smallest chain
            best = 10000
            for l in range(L):
                if drawn[g, l] == 0:
                    c = 0
                    for j in range(2):
                        b = line_box[l, j]
                        if b >= 0 and sides[g, b] == 2:
                            c += 1
                    if c < best:
                        best = c
                        pick = l
        out[g] = pick


# ------------------------------------------------------------------- wrapper

class Batch:
    """G games played in lockstep."""

    def __init__(self, geo, G, rng=None):
        self.geo, self.G = geo, G
        self.rng = rng or np.random.default_rng()
        L, B = geo.L, geo.B
        self.drawn = np.zeros((G, L), np.uint8)
        self.sides = np.zeros((G, B), np.int8)
        self.owner = np.zeros((G, B), np.int8)
        self.score = np.zeros((G, 2), np.int32)
        self.mover = np.ones(G, np.int8)
        self.done = np.zeros(G, np.bool_)
        self._comp = np.zeros((G, B), np.int64)
        self._size = np.zeros((G, B + 1), np.int64)
        self._formed = np.zeros((G, B + 1), np.int64)
        self._ground = np.zeros((G, B + 1), np.int64)
        self._stack = np.zeros((G, B + 1), np.int64)
        self._hbuf = np.zeros(G, np.int64)

    def reset(self, idx=None):
        idx = slice(None) if idx is None else idx
        self.drawn[idx] = 0
        self.sides[idx] = 0
        self.owner[idx] = 0
        self.score[idx] = 0
        self.mover[idx] = 1
        self.done[idx] = False

    def clone_state(self):
        return (self.drawn.copy(), self.owner.copy(), self.score.copy(),
                self.mover.copy(), self.done.copy())

    def legal(self):
        return self.drawn == 0

    def step(self, actions):
        caps, same = _apply(self.drawn, self.sides, self.owner, self.score,
                            self.mover, self.done, np.ascontiguousarray(actions, np.int64),
                            self.geo.line_box, self.geo.B)
        return caps / self.geo.B, same.astype(bool)

    def margin(self):
        """Final margin from player 1's point of view."""
        return self.score[:, 0] - self.score[:, 1]

    def heuristic(self):
        _heuristic(self.drawn, self.sides, self.geo.line_box, self.geo.box_lines,
                   self.geo.L, self.geo.B, self.done, self._hbuf)
        return self._hbuf.copy()

    # ---- observation ----
    def encode(self, out=None):
        if not hasattr(self, "_enc"):
            self._enc = Encoder(self.geo, self.G)
        return self._enc(self.drawn, self.sides, self.owner, self.score, self.mover, out)


class Encoder:
    """Turns raw board arrays into network input. Reused for self-play and for
    replay batches, so training and acting always see identical features."""

    def __init__(self, geo, capacity):
        self.geo = geo
        B = geo.B
        self.cap = capacity
        self._comp = np.zeros((capacity, B), np.int64)
        self._size = np.zeros((capacity, B + 1), np.int64)
        self._formed = np.zeros((capacity, B + 1), np.int64)
        self._ground = np.zeros((capacity, B + 1), np.int64)
        self._stack = np.zeros((capacity, B + 1), np.int64)

    def __call__(self, drawn, sides, owner, score, mover, out=None):
        """(G, C_IN, S, S) float32.

        0  on board            1  line drawn          2  line undrawn
        3  line completes a box                       4  line gives a box away
        5..8  box has 0/1/2/3 sides                   9  box owned by mover
        10 box owned by opponent                     11 dots
        12 score difference / B (mover's view)       13 lines left / L
        14 safe lines left / L                       15 parity of lines left
        -- chain structure, the part a plain conv net cannot work out itself --
        16 size of this box's component / B          17 component is fully formed
        18 long chains on the board / B              19 PARITY of that count
        """
        geo = self.geo
        S, L, B = geo.S, geo.L, geo.B
        G = drawn.shape[0]
        assert G <= self.cap, "encoder scratch too small"
        X = np.zeros((G, C_IN, S, S), np.float32) if out is None else out[:G]
        X[:] = 0.0
        lr, lc = geo.line_rc
        br, bc = geo.box_rc

        X[:, 0] = 1.0
        X[:, 11, ::2, ::2] = 1.0
        X[:, 1][:, lr, lc] = drawn
        X[:, 2][:, lr, lc] = 1 - drawn

        s_of_line = sides[:, geo.line_box]
        valid = geo.line_box[None, :, :] >= 0
        cap = ((s_of_line == 3) & valid).any(2) & (drawn == 0)
        giv = ((s_of_line == 2) & valid).any(2) & (drawn == 0)
        X[:, 3][:, lr, lc] = cap
        X[:, 4][:, lr, lc] = giv
        safe = (drawn == 0) & ~cap & ~giv

        for k in range(4):
            X[:, 5 + k][:, br, bc] = (sides == k)
        mine = owner == mover[:, None]
        X[:, 9][:, br, bc] = (sides == 4) & mine
        X[:, 10][:, br, bc] = (sides == 4) & ~mine & (owner != 0)

        diff = (score[:, 0].astype(np.float32) - score[:, 1])
        diff = np.where(mover == 2, -diff, diff) / B
        rem = (drawn == 0).sum(1).astype(np.float32)
        X[:, 12] = diff[:, None, None]
        X[:, 13] = (rem / L)[:, None, None]
        X[:, 14] = (safe.sum(1) / L).astype(np.float32)[:, None, None]
        X[:, 15] = (rem % 2)[:, None, None]

        nc = _components(np.ascontiguousarray(drawn), np.ascontiguousarray(sides),
                         geo.line_box, geo.box_lines, B, self._comp[:G], self._size[:G],
                         self._formed[:G], self._ground[:G], self._stack[:G])
        open_ = sides < 4
        gi = np.arange(G)[:, None]
        cid = np.where(open_, self._comp[:G], 0)
        X[:, 16][:, br, bc] = np.where(open_, self._size[:G][gi, cid] / B, 0.0)
        X[:, 17][:, br, bc] = np.where(open_, self._formed[:G][gi, cid], 0)

        nlong = np.zeros(G, np.float32)
        for g in range(G):
            k = nc[g]
            if k:
                sz = self._size[g, :k]
                fm = self._formed[g, :k]
                gr = self._ground[g, :k]
                nlong[g] = int(((fm == 1) & (gr == 2) & (sz >= 3)).sum())
        X[:, 18] = (nlong / B)[:, None, None]
        X[:, 19] = (nlong % 2)[:, None, None]
        return X


def sides_from_drawn(geo, drawn):
    """Recover per-box drawn-side counts from a batch of line vectors."""
    return drawn[:, geo.box_lines].sum(2).astype(np.int8)
