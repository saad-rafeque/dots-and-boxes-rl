#!/usr/bin/env python3
# =====================================================================
#  NIGHTMARE DOTS & BOXES  v3
#  AlphaZero-style RL  +  batched MCTS  +  exact alpha-beta endgame solver
#  + chain theory (exact simple-loony endgames) + multi-threaded solver
#  + loss book (AI remembers the games it lost) + terminal play + GUI (dab_gui.py)
#
#  Boards are counted in BOXES:  --board 5 -> 5x5 boxes (60 lines)
#                                --board 6 -> 6x6 boxes (84 lines)
#
#  Commands
#    python nightmare_dab.py test [--quick]                correctness tests
#    python nightmare_dab.py bench --board 6               speed benchmark
#    python nightmare_dab.py train --board 5 --out runs/b5 --hours 9.4
#    python nightmare_dab.py play  --board 6 --model models/best_6x6.pt --think 8
#    python nightmare_dab.py learn --board 6 --model models/best_6x6.pt
#    python dab_gui.py                                     window: human vs AI (5x5 / 6x6)
#
#  KAGGLE: open dab_kaggle.ipynb, Accelerator = GPU T4 x2, then
#    Save Version -> "Save & Run All (Commit)". It runs in the background
#    (browser/laptop can be closed), trains 5x5 on GPU0 and 6x6 on GPU1,
#    checkpoints every 15 min and stops by itself before 9.5 h.
#  Next session: add the previous notebook's output as input; the launcher
#    finds runs/b5 and runs/b6 there and resumes automatically.
#
#  Windows laptop (no GPU needed to play):  pip install numpy numba torch
# =====================================================================

import os
import sys
import time
import math
import json
import pickle
import random
import shutil
import queue
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
from numba import njit, types
from numba.typed import Dict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # the solver/tests for the game core still work without torch
    torch = None


# =====================================================================
# 1. BOARD GEOMETRY  (runtime N, so one process can handle any size)
# =====================================================================
#
#  Grid of size S = 2N+1:
#    dots  at (even, even)      horizontal lines at (even, odd)
#    boxes at (odd,  odd)       vertical lines   at (odd,  even)
#  All 8 symmetries are done on this grid, so they are correct by construction.

BASE_KEY = np.uint64(0x9E3779B97F4A7C15)


class Geometry:
    def __init__(self, N):
        assert 2 <= N <= 7, "board must be 2..7 boxes per side"
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
                box_lines[b] = (self.h(r, c), self.h(r + 1, c), self.v(r, c), self.v(r, c + 1))

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
        line_inv = np.zeros_like(line_perm)
        box_inv = np.zeros_like(box_perm)
        for t in range(8):
            line_inv[t, line_perm[t]] = np.arange(L)
            box_inv[t, box_perm[t]] = np.arange(B)

        rng = np.random.default_rng(20240917 + N)
        self.zob = rng.integers(1, np.iinfo(np.uint64).max, size=L, dtype=np.uint64)

        self.line_cell, self.box_cell = line_cell, box_cell
        self.box_lines, self.line_box = box_lines, line_box
        self.cell_line, self.cell_box = cell_line, cell_box
        self.line_perm, self.box_perm = line_perm, box_perm
        self.line_inv, self.box_inv = line_inv, box_inv
        self.line_flat = (line_cell[:, 0] * S + line_cell[:, 1]).astype(np.int64)
        self.box_flat = (box_cell[:, 0] * S + box_cell[:, 1]).astype(np.int64)

    def h(self, r, c):  # horizontal line from dot (r,c) to (r,c+1)
        return r * self.N + c

    def v(self, r, c):  # vertical line from dot (r,c) to (r+1,c)
        return self.NH + r * (self.N + 1) + c

    def _tf(self, t, r, c):
        m = self.S - 1
        return [(r, c), (c, m - r), (m - r, m - c), (m - c, r),
                (r, m - c), (m - r, c), (c, r), (m - c, m - r)][t]

    def describe_line(self, l):
        if l < self.NH:
            return "h %d %d" % divmod(l, self.N)
        r, c = divmod(l - self.NH, self.N + 1)
        return "v %d %d" % (r, c)

    def key_of(self, drawn):
        k = BASE_KEY
        for l in np.nonzero(drawn)[0]:
            k ^= self.zob[l]
        return k

    def mask_of(self, drawn):
        return int(sum(1 << int(l) for l in np.nonzero(drawn)[0]))

    def canonical(self, drawn):
        """Return (canonical bitmask, transform t) over the 8 symmetries."""
        idx = np.nonzero(drawn)[0]
        best, bt = None, 0
        for t in range(8):
            m = int(sum(1 << int(x) for x in self.line_perm[t, idx]))
            if best is None or m < best:
                best, bt = m, t
        return best, bt


GEO_CACHE = {}


def geometry(N):
    if N not in GEO_CACHE:
        GEO_CACHE[N] = Geometry(N)
    return GEO_CACHE[N]


# =====================================================================
# 2. NUMBA GAME CORE
# =====================================================================
#  drawn[l] uint8 (0/1), sides[b] int8 (drawn sides 0..4), owner[b] int8 (0,1,2)
#  Rule: completing a box -> same player moves again.

@njit(cache=True)
def nb_seed(s):
    np.random.seed(s)


@njit(cache=True)
def apply_move(drawn, sides, owner, line_box, m, mover):
    drawn[m] = 1
    k = 0
    for j in range(2):
        b = line_box[m, j]
        if b >= 0:
            sides[b] += 1
            if sides[b] == 4:
                owner[b] = mover
                k += 1
    return k


@njit(cache=True)
def count_uncaptured(sides, B):
    n = 0
    for b in range(B):
        if sides[b] < 4:
            n += 1
    return n


@njit(cache=True)
def count_undrawn(drawn, L):
    n = 0
    for l in range(L):
        if drawn[l] == 0:
            n += 1
    return n


@njit(cache=True)
def count_safe_lines(drawn, sides, line_box, L):
    """Lines that neither capture nor hand the opponent a box."""
    n = 0
    for l in range(L):
        if drawn[l] == 0:
            ok = True
            for j in range(2):
                b = line_box[l, j]
                if b >= 0 and sides[b] >= 2:
                    ok = False
            if ok:
                n += 1
    return n


@njit(cache=True)
def util(margin, B):
    """Utility in [-1,1]: winning matters most, margin second. Monotonic in margin."""
    s = 0.0
    if margin > 0:
        s = 1.0
    elif margin < 0:
        s = -1.0
    u = 0.5 * s + 0.5 * margin / B
    if u > 1.0:
        u = 1.0
    if u < -1.0:
        u = -1.0
    return u


# =====================================================================
# 3. EXACT SOLVER  (negamax alpha-beta + chain theory, score-independent TT)
# =====================================================================
#  Value = net boxes the player-to-move wins from the REMAINING boxes.
#  It depends only on the drawn lines, so the TT key is the line hash only.
#
#  `flags` bits (all verified against brute force / plain search in `test`):
#   1  dominance pruning
#      (a) a capturable box whose capture does not create another capturable box
#          -> capture it, nothing else needs to be searched.
#      (b) capturable boxes exist but all captures are "unsafe" (inside a chain)
#          -> only capture lines and the double-dealing decline lines matter.
#   2  chain de-duplication: inside an isolated chain or loop every opening line
#      is equivalent (2-chains keep one end line + the middle line).
#   4  simple-loony-endgame theory: when every open box has exactly 2 sides the
#      board is a set of independent chains and loops, and its exact value comes
#      from a tiny memoised recursion instead of a search:
#         open 1- or 2-chain  : -(n + V')
#         open chain  n >= 3  : -(n - 2 + |V' + 2|)
#         open loop   n >= 4  : -(n - 4 + |V' + 4|)
#      (V' = value of the rest for whoever has to move in it)

EXACT, LOWER, UPPER = 1, 2, 3
F_PRUNE, F_DEDUP, F_THEORY = 1, 2, 4
F_ALL = 7
N_CTYPES = 128          # component type t: t < 64 -> chain of t boxes, t >= 64 -> loop of t-64 boxes


@njit(cache=True)
def _undrawn_of_box(drawn, box_lines, b):
    for j in range(4):
        l = box_lines[b, j]
        if drawn[l] == 0:
            return l
    return -1


@njit(cache=True)
def _other_box(line_box, l, b):
    b0 = line_box[l, 0]
    if b0 == b:
        return line_box[l, 1]
    return b0


@njit(cache=True)
def _add_unique(buf, n, m):
    for i in range(n):
        if buf[i] == m:
            return n
    buf[n] = m
    return n + 1


@njit(cache=True)
def label_components(drawn, sides, line_box, box_lines, B, W):
    """Connected components of open boxes (linked by undrawn lines).
    W[0,b] = component id (-1 captured), W[1,c] = size, W[2,c] = 1 if all boxes have
    exactly 2 sides, W[3,c] = number of undrawn lines to the ground, W[4] = BFS stack.
    Returns the number of components."""
    for b in range(B):
        W[0, b] = -1
    nc = 0
    for s in range(B):
        if sides[s] >= 4 or W[0, s] >= 0:
            continue
        W[0, s] = nc
        W[1, nc] = 0
        W[2, nc] = 1
        W[3, nc] = 0
        top = 0
        W[4, top] = s
        top += 1
        while top > 0:
            top -= 1
            b = W[4, top]
            W[1, nc] += 1
            if sides[b] != 2:
                W[2, nc] = 0
            for j in range(4):
                l = box_lines[b, j]
                if drawn[l] != 0:
                    continue
                o = _other_box(line_box, l, b)
                if o < 0:
                    W[3, nc] += 1
                elif W[0, o] < 0:
                    W[0, o] = nc
                    W[4, top] = o
                    top += 1
        nc += 1
    return nc


@njit
def _loony_dp(cnt, key, memo, R):
    if key in memo:
        return memo[key]
    best = -1000
    anyc = False
    for t in range(1, N_CTYPES):
        if cnt[t] == 0:
            continue
        anyc = True
        cnt[t] -= 1
        vp = _loony_dp(cnt, key ^ R[t, cnt[t]], memo, R)
        cnt[t] += 1
        if t < 64:
            n = t
            if n <= 2:
                v = -(n + vp)
            else:
                v = -(n - 2 + abs(vp + 2))
        else:
            n = t - 64
            v = -(n - 4 + abs(vp + 4))
        if v > best:
            best = v
    if not anyc:
        best = 0
    memo[key] = best
    return best


@njit
def simple_value(drawn, sides, line_box, box_lines, B, W, cnt, memo, R):
    """Exact value if the position is a simple loony endgame, else -1000."""
    for b in range(B):
        if sides[b] < 2 or sides[b] == 3:
            return -1000
    nc = label_components(drawn, sides, line_box, box_lines, B, W)
    for t in range(N_CTYPES):
        cnt[t] = 0
    key = np.uint64(0)
    for c in range(nc):
        n = W[1, c]
        g = W[3, c]
        if g == 2:
            t = n
        elif g == 0:
            t = 64 + n
        else:
            return -1000
        key ^= R[t, cnt[t]]
        cnt[t] += 1
    if len(memo) > 4_000_000:
        memo.clear()
    return _loony_dp(cnt, key, memo, R)


@njit(cache=True)
def gen_moves(drawn, sides, line_box, box_lines, L, B, buf, tt_move, flags, W, seen):
    n = 0
    if flags & F_PRUNE:
        any_cap = False
        for b in range(B):
            if sides[b] == 3:
                any_cap = True
                l = _undrawn_of_box(drawn, box_lines, b)
                o = _other_box(line_box, l, b)
                if o < 0 or sides[o] != 2:
                    buf[0] = l
                    return 1
        if any_cap:
            for b in range(B):
                if sides[b] == 3:
                    l = _undrawn_of_box(drawn, box_lines, b)
                    n = _add_unique(buf, n, l)
                    o = _other_box(line_box, l, b)
                    for j in range(4):
                        l2 = box_lines[o, j]
                        if drawn[l2] == 0 and l2 != l:
                            n = _add_unique(buf, n, l2)
            if tt_move >= 0:
                for i in range(n):
                    if buf[i] == tt_move:
                        buf[i] = buf[0]
                        buf[0] = tt_move
                        break
            return n
    dedup = False
    if flags & F_DEDUP:
        # only useful (and only valid) when nothing is capturable
        dedup = True
        for b in range(B):
            if sides[b] == 3:
                dedup = False
                break
        if dedup:
            nc = label_components(drawn, sides, line_box, box_lines, B, W)
            for c in range(nc):
                seen[c] = 0          # bit0: end/any line used, bit1: middle line used (2-chains)
    if tt_move >= 0 and drawn[tt_move] == 0:
        buf[n] = tt_move
        n += 1
        if dedup:
            b0 = line_box[tt_move, 0]
            c = W[0, b0]
            if W[2, c] == 1:
                if W[1, c] == 2 and W[3, c] == 2 and line_box[tt_move, 1] >= 0:
                    seen[c] |= 2
                else:
                    seen[c] |= 1
    for pas in range(3):  # 0: captures, 1: safe moves, 2: moves that give boxes
        for l in range(L):
            if drawn[l] != 0 or l == tt_move:
                continue
            cap = False
            uns = False
            for j in range(2):
                b = line_box[l, j]
                if b >= 0:
                    if sides[b] == 3:
                        cap = True
                    elif sides[b] == 2:
                        uns = True
            if pas == 0 and cap:
                buf[n] = l
                n += 1
            elif pas == 1 and (not cap) and (not uns):
                buf[n] = l
                n += 1
            elif pas == 2 and (not cap) and uns:
                if dedup:
                    c = W[0, line_box[l, 0]]
                    if W[2, c] == 1:
                        bit = 1
                        if W[1, c] == 2 and W[3, c] == 2 and line_box[l, 1] >= 0:
                            bit = 2
                        if seen[c] & bit:
                            continue
                        seen[c] |= bit
                buf[n] = l
                n += 1
    return n


@njit
def _ab(drawn, sides, line_box, box_lines, L, B, zob, key, rb, alpha, beta,
        tt_key, tt_val, tt_flag, tt_move, tt_mask, budget, ply, mbuf, flags, W, SEEN, cnt, memo, R):
    if rb == 0:
        return 0
    budget[0] -= 1
    if budget[0] < 0:
        return 0
    if beta <= -rb:
        return -rb
    if alpha >= rb:
        return rb
    if flags & F_THEORY:
        sv = simple_value(drawn, sides, line_box, box_lines, B, W, cnt, memo, R)
        if sv > -1000:
            return sv
    idx = np.int64(key & tt_mask)
    ttm = -1
    if tt_key[idx] == key:
        v = np.int64(tt_val[idx])
        f = tt_flag[idx]
        if f == EXACT:
            return v
        if f == LOWER and v >= beta:
            return v
        if f == UPPER and v <= alpha:
            return v
        ttm = np.int64(tt_move[idx])
    n = gen_moves(drawn, sides, line_box, box_lines, L, B, mbuf[ply], ttm, flags, W, SEEN[ply])
    a0 = alpha
    best = -1000
    bestm = np.int64(mbuf[ply, 0])
    for i in range(n):
        m = np.int64(mbuf[ply, i])
        drawn[m] = 1
        k = 0
        for j in range(2):
            b = line_box[m, j]
            if b >= 0:
                sides[b] += 1
                if sides[b] == 4:
                    k += 1
        nk = key ^ zob[m]
        if k > 0:
            v = k + _ab(drawn, sides, line_box, box_lines, L, B, zob, nk, rb - k,
                        alpha - k, beta - k, tt_key, tt_val, tt_flag, tt_move, tt_mask,
                        budget, ply + 1, mbuf, flags, W, SEEN, cnt, memo, R)
        else:
            v = -_ab(drawn, sides, line_box, box_lines, L, B, zob, nk, rb,
                     -beta, -alpha, tt_key, tt_val, tt_flag, tt_move, tt_mask,
                     budget, ply + 1, mbuf, flags, W, SEEN, cnt, memo, R)
        drawn[m] = 0
        for j in range(2):
            b = line_box[m, j]
            if b >= 0:
                sides[b] -= 1
        if budget[0] < 0:
            return 0
        if v > best:
            best = v
            bestm = m
            if v > alpha:
                alpha = v
                if alpha >= beta:
                    break
    if best <= a0:
        f = UPPER
    elif best >= beta:
        f = LOWER
    else:
        f = EXACT
    tt_key[idx] = key
    tt_val[idx] = best
    tt_flag[idx] = f
    tt_move[idx] = bestm
    return best


@njit(nogil=True)
def solve_root(drawn, sides, line_box, box_lines, L, B, zob, key,
               tt_key, tt_val, tt_flag, tt_move, tt_mask, budget, mbuf, out_moves, flags,
               W, SEEN, cnt, memo, R):
    """Returns (ok, best_value, n_best). out_moves[:n_best] = ALL optimal moves
    (root moves are never de-duplicated, only pruned by dominance)."""
    rb = count_uncaptured(sides, B)
    if rb == 0:
        return 1, 0, 0
    n = gen_moves(drawn, sides, line_box, box_lines, L, B, mbuf[0], -1, flags & F_PRUNE, W, SEEN[0])
    best = -1000
    nb = 0
    for i in range(n):
        m = np.int64(mbuf[0, i])
        drawn[m] = 1
        k = 0
        for j in range(2):
            b = line_box[m, j]
            if b >= 0:
                sides[b] += 1
                if sides[b] == 4:
                    k += 1
        lo = best - 1
        if best <= -1000:
            lo = -1000
        nk = key ^ zob[m]
        if k > 0:
            v = k + _ab(drawn, sides, line_box, box_lines, L, B, zob, nk, rb - k,
                        lo - k, rb + 1 - k, tt_key, tt_val, tt_flag, tt_move, tt_mask,
                        budget, 1, mbuf, flags, W, SEEN, cnt, memo, R)
        else:
            v = -_ab(drawn, sides, line_box, box_lines, L, B, zob, nk, rb,
                     -(rb + 1), -lo, tt_key, tt_val, tt_flag, tt_move, tt_mask,
                     budget, 1, mbuf, flags, W, SEEN, cnt, memo, R)
        drawn[m] = 0
        for j in range(2):
            b = line_box[m, j]
            if b >= 0:
                sides[b] -= 1
        if budget[0] < 0:
            return 0, 0, 0
        if v > best:
            best = v
            nb = 0
            out_moves[nb] = m
            nb += 1
        elif v == best:
            out_moves[nb] = m
            nb += 1
    return 1, best, nb


def new_loony_memo():
    return Dict.empty(key_type=types.uint64, value_type=types.int64)


LOONY_R = np.random.default_rng(777).integers(1, np.iinfo(np.uint64).max, size=(N_CTYPES, 64), dtype=np.uint64)


class Solver:
    """Owns a transposition table + scratch for one board size (reused across positions).
    One Solver per thread."""

    def __init__(self, geo, tt_bits=22):
        self.geo = geo
        n = 1 << tt_bits
        self.tt_key = np.zeros(n, np.uint64)
        self.tt_val = np.zeros(n, np.int8)
        self.tt_flag = np.zeros(n, np.int8)
        self.tt_move = np.zeros(n, np.int16)
        self.tt_mask = np.uint64(n - 1)
        self.mbuf = np.zeros((geo.L + 2, geo.L), np.int16)
        self.W = np.zeros((5, geo.B + 1), np.int64)
        self.SEEN = np.zeros((geo.L + 2, geo.B + 1), np.int64)
        self.cnt = np.zeros(N_CTYPES, np.int64)
        self.memo = new_loony_memo()
        self.budget = np.zeros(1, np.int64)
        self.out = np.zeros(geo.L, np.int64)
        self.nodes_used = 0

    def solve(self, drawn, sides, budget=2_000_000, prune=True):
        """Exact solve. Returns (ok, value_for_mover, optimal_moves).
        prune: True -> all speedups, False -> plain alpha-beta, or an int flag mask."""
        flags = F_ALL if prune is True else (0 if prune is False else int(prune))
        g = self.geo
        d = np.ascontiguousarray(drawn, dtype=np.uint8).copy()
        s = np.ascontiguousarray(sides, dtype=np.int8).copy()
        self.budget[0] = budget
        ok, val, nb = solve_root(d, s, g.line_box, g.box_lines, g.L, g.B, g.zob, g.key_of(d),
                                 self.tt_key, self.tt_val, self.tt_flag, self.tt_move,
                                 self.tt_mask, self.budget, self.mbuf, self.out, flags,
                                 self.W, self.SEEN, self.cnt, self.memo, LOONY_R)
        self.nodes_used = budget - max(0, int(self.budget[0]))
        if not ok:
            return False, 0, []
        return True, int(val), [int(x) for x in self.out[:nb]]

    def clear(self):
        self.tt_key[:] = 0


# =====================================================================
# 4. HEURISTIC BOT  (baseline opponent for evaluation)
# =====================================================================
#  Takes boxes greedily, plays safe moves while they exist, otherwise gives
#  away the fewest boxes. Roughly a decent casual human.

@njit(cache=True)
def _greedy_loss_after(drawn, sides, line_box, box_lines, L, B, m, stack):
    ns = 0
    drawn[m] = 1
    for j in range(2):
        b = line_box[m, j]
        if b >= 0:
            sides[b] += 1
    stack[ns] = m
    ns += 1
    given = 0
    found = True
    while found:
        found = False
        for b in range(B):
            if sides[b] == 3:
                l = _undrawn_of_box(drawn, box_lines, b)
                drawn[l] = 1
                for j in range(2):
                    bb = line_box[l, j]
                    if bb >= 0:
                        sides[bb] += 1
                        if sides[bb] == 4:
                            given += 1
                stack[ns] = l
                ns += 1
                found = True
                break
    for i in range(ns - 1, -1, -1):
        l = stack[i]
        drawn[l] = 0
        for j in range(2):
            bb = line_box[l, j]
            if bb >= 0:
                sides[bb] -= 1
    return given


@njit(cache=True)
def heuristic_move(drawn, sides, line_box, box_lines, L, B, stack, cand):
    for b in range(B):
        if sides[b] == 3:
            l = _undrawn_of_box(drawn, box_lines, b)
            o = _other_box(line_box, l, b)
            if o < 0 or sides[o] != 2:
                return l
    for b in range(B):
        if sides[b] == 3:
            return _undrawn_of_box(drawn, box_lines, b)
    ns = 0
    for l in range(L):
        if drawn[l] == 0:
            uns = False
            for j in range(2):
                b = line_box[l, j]
                if b >= 0 and sides[b] == 2:
                    uns = True
            if not uns:
                cand[ns] = l
                ns += 1
    if ns > 0:
        return cand[np.random.randint(ns)]
    best = 100000
    nb = 0
    for l in range(L):
        if drawn[l] == 0:
            c = _greedy_loss_after(drawn, sides, line_box, box_lines, L, B, l, stack)
            if c < best:
                best = c
                nb = 0
                cand[nb] = l
                nb += 1
            elif c == best:
                cand[nb] = l
                nb += 1
    return cand[np.random.randint(nb)]




# =====================================================================
# 5. NETWORK INPUT ENCODING  (18 planes on the (2N+1)x(2N+1) grid)
# =====================================================================
#  0 on-board   1 drawn line   2 undrawn line   3 line captures a box
#  4 line gives a box away     5-8 box has 0/1/2/3 sides drawn
#  9 box owned by player to move   10 box owned by opponent   11 dots
#  12 score diff/B   13 lines left/L   14 safe lines/L   15 boxes left/B
#  16 parity of lines left          17 parity of safe lines
C_IN = 18


@njit(cache=True)
def encode_state(X, i, drawn, sides, owner, mover, diff, line_box, line_cell, box_cell, L, B):
    S = X.shape[2]
    for c in range(X.shape[1]):
        for r in range(S):
            for q in range(S):
                X[i, c, r, q] = 0.0
    rem = 0
    nsafe = 0
    rb = 0
    for l in range(L):
        r = line_cell[l, 0]
        q = line_cell[l, 1]
        if drawn[l] != 0:
            X[i, 1, r, q] = 1.0
        else:
            rem += 1
            X[i, 2, r, q] = 1.0
            cap = False
            uns = False
            for j in range(2):
                b = line_box[l, j]
                if b >= 0:
                    if sides[b] == 3:
                        cap = True
                    elif sides[b] == 2:
                        uns = True
            if cap:
                X[i, 3, r, q] = 1.0
            if uns:
                X[i, 4, r, q] = 1.0
            if (not cap) and (not uns):
                nsafe += 1
    for b in range(B):
        r = box_cell[b, 0]
        q = box_cell[b, 1]
        s = np.int64(sides[b])
        if s < 4:
            rb += 1
            X[i, 5 + s, r, q] = 1.0
        elif owner[b] == mover:
            X[i, 9, r, q] = 1.0
        else:
            X[i, 10, r, q] = 1.0
    gd = diff / B
    gr = rem / L
    gs = nsafe / L
    gb = rb / B
    pr = float(rem % 2)
    ps = float(nsafe % 2)
    for r in range(S):
        for q in range(S):
            X[i, 0, r, q] = 1.0
            if r % 2 == 0 and q % 2 == 0:
                X[i, 11, r, q] = 1.0
            X[i, 12, r, q] = gd
            X[i, 13, r, q] = gr
            X[i, 14, r, q] = gs
            X[i, 15, r, q] = gb
            X[i, 16, r, q] = pr
            X[i, 17, r, q] = ps


@njit(cache=True)
def encode_batch(X, drawn_b, owner_b, s_b, mover_b, line_box, box_lines, line_cell, box_cell, L, B):
    n = drawn_b.shape[0]
    sides = np.zeros(B, np.int8)
    for i in range(n):
        for b in range(B):
            c = 0
            for j in range(4):
                c += drawn_b[i, box_lines[b, j]]
            sides[b] = c
        mv = np.int64(mover_b[i])
        diff = np.int64(s_b[i, 0]) - np.int64(s_b[i, 1])
        if mv == 2:
            diff = -diff
        encode_state(X, i, drawn_b[i], sides, owner_b[i], mv, diff, line_box, line_cell, box_cell, L, B)


# =====================================================================
# 6. NEURAL NETWORK  (fully convolutional: same weights run on 5x5 and 6x6)
# =====================================================================

if torch is not None:
    class GlobalPoolBias(nn.Module):
        """KataGo-style global pooling: lets every cell see board-wide parity/score."""

        def __init__(self, ch):
            super().__init__()
            self.fc = nn.Linear(2 * ch, ch)

        def forward(self, x):
            g = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
            return self.fc(g)[:, :, None, None]

    class ResBlock(nn.Module):
        def __init__(self, ch, gpool):
            super().__init__()
            self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.b1 = nn.BatchNorm2d(ch)
            self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.b2 = nn.BatchNorm2d(ch)
            self.gp = GlobalPoolBias(ch) if gpool else None

        def forward(self, x):
            y = F.relu(self.b1(self.c1(x)))
            if self.gp is not None:
                y = y + self.gp(y)
            y = self.b2(self.c2(y))
            return F.relu(x + y)

    class DABNet(nn.Module):
        def __init__(self, channels=64, blocks=8):
            super().__init__()
            self.channels, self.blocks = channels, blocks
            self.stem = nn.Sequential(nn.Conv2d(C_IN, channels, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(channels), nn.ReLU())
            self.trunk = nn.Sequential(*[ResBlock(channels, gpool=(i % 3 == 2)) for i in range(blocks)])
            self.pol = nn.Sequential(nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                                     nn.Conv2d(32, 1, 1))
            self.own = nn.Conv2d(channels, 1, 1)
            self.val_conv = nn.Sequential(nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
            self.val_fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, x, line_flat, box_flat):
            h = self.trunk(self.stem(x))
            n = x.shape[0]
            p = self.pol(h).reshape(n, -1).index_select(1, line_flat)
            o = torch.tanh(self.own(h).reshape(n, -1).index_select(1, box_flat))
            vh = self.val_conv(h)
            vg = torch.cat([vh.mean(dim=(2, 3)), vh.amax(dim=(2, 3))], dim=1)
            v = torch.tanh(self.val_fc(vg)).squeeze(1)
            return p, v, o

    class NetEvaluator:
        def __init__(self, model, geo, device):
            self.model, self.geo, self.device = model, geo, device
            self.amp = device.type == "cuda"
            self.cl = device.type == "cuda"      # channels_last -> faster fp16 convs on T4/P100
            self.line_flat = torch.as_tensor(geo.line_flat, device=device)
            self.box_flat = torch.as_tensor(geo.box_flat, device=device)
            self.pin = None

        def to_device(self, X):
            X = np.ascontiguousarray(X)
            if self.device.type != "cuda":
                return torch.from_numpy(X)
            n = X.shape[0]
            if self.pin is None or self.pin.shape[0] < n or self.pin.shape[1:] != X.shape[1:]:
                self.pin = torch.empty((max(n, 1024),) + X.shape[1:], dtype=torch.float32, pin_memory=True)
            buf = self.pin[:n]
            buf.copy_(torch.from_numpy(X))
            x = buf.to(self.device, non_blocking=True)
            return x.contiguous(memory_format=torch.channels_last)

        @torch.inference_mode()
        def __call__(self, X):
            x = self.to_device(X)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                p, v, _ = self.model(x, self.line_flat, self.box_flat)
            return p.float().cpu().numpy(), v.float().cpu().numpy()


# =====================================================================
# 7. BATCHED MCTS  (numba; many independent trees, one GPU batch per round)
# =====================================================================
#  Node storage per game g, node n:
#    TI[g,n] = parent, child_start, visits, virtual_loss      (int32)
#    TF[g,n] = prior, value_sum (node mover's view), exact value (float32)
#    TS[g,n] = move, n_children (int16)    TB[g,n] = mover, state (int8)

I_PAR, I_CST, I_N, I_VL = 0, 1, 2, 3
F_P, F_W, F_V = 0, 1, 2
S_MOVE, S_NCH = 0, 1
B_MOVER, B_STATE = 0, 1
ST_UNEXP, ST_EXP, ST_TERM, ST_SOLVED, ST_PEND, ST_LEAF = 0, 1, 2, 3, 4, 5


@njit(cache=True)
def tree_reset(TI, TF, TS, TB, t_used, g, root_mover):
    t_used[g] = 1
    TI[g, 0, I_PAR] = -1
    TI[g, 0, I_CST] = 0
    TI[g, 0, I_N] = 0
    TI[g, 0, I_VL] = 0
    TF[g, 0, F_P] = 1.0
    TF[g, 0, F_W] = 0.0
    TF[g, 0, F_V] = 0.0
    TS[g, 0, S_MOVE] = -1
    TS[g, 0, S_NCH] = 0
    TB[g, 0, B_MOVER] = root_mover
    TB[g, 0, B_STATE] = ST_UNEXP


@njit(cache=True)
def _backup(TI, TF, TB, g, node, v):
    v1 = v
    if TB[g, node, B_MOVER] != 1:
        v1 = -v
    x = node
    while x >= 0:
        TI[g, x, I_N] += 1
        if TB[g, x, B_MOVER] == 1:
            TF[g, x, F_W] += v1
        else:
            TF[g, x, F_W] -= v1
        x = TI[g, x, I_PAR]


@njit(cache=True)
def _vl_path(TI, g, node, delta):
    x = node
    while x >= 0:
        TI[g, x, I_VL] += delta
        x = TI[g, x, I_PAR]


@njit(cache=True)
def _puct_child(TI, TF, TS, TB, g, p, c_puct, fpu_red):
    mp = TB[g, p, B_MOVER]
    npv = TI[g, p, I_N] + TI[g, p, I_VL]
    sq = math.sqrt(max(1.0, float(npv)))
    pq = 0.0
    if TI[g, p, I_N] > 0:
        pq = TF[g, p, F_W] / TI[g, p, I_N]
    fpu = pq - fpu_red
    s = TI[g, p, I_CST]
    e = s + TS[g, p, S_NCH]
    best = -1e30
    bi = s
    for c in range(s, e):
        vlc = TI[g, c, I_VL]
        n = TI[g, c, I_N] + vlc
        if n > 0:
            w = TF[g, c, F_W]
            if TB[g, c, B_MOVER] != mp:
                w = -w
            q = (w - vlc) / n
        else:
            q = fpu
        u = q + c_puct * TF[g, c, F_P] * sq / (1.0 + n)
        if u > best:
            best = u
            bi = c
    return bi


@njit  # calls the recursive solver -> not cached (numba cache + recursion segfaults)
def mcts_select(TI, TF, TS, TB, t_used, R_drawn, R_sides, R_owner, R_info, R_key, ctl,
                SC_drawn, SC_sides, SC_owner, LF_info, LF_drawn, LF_sides, X,
                line_box, box_lines, line_cell, box_cell, zob,
                tt_key, tt_val, tt_flag, tt_move, tt_mask, budget, mbuf, prm,
                W, SEEN, cnt, memo, R):
    c_puct = prm[0]
    fpu_red = prm[1]
    leaf_solve = np.int64(prm[2])
    leaf_budget = np.int64(prm[3])
    K = np.int64(prm[4])
    vl = np.int64(prm[5])
    max_att = np.int64(prm[6])
    G = R_drawn.shape[0]
    L = R_drawn.shape[1]
    B = R_sides.shape[1]
    nslot = 0
    for g in range(G):
        if ctl[g, 0] == 0:
            continue
        got = 0
        att = 0
        while got < K and att < max_att and ctl[g, 1] < ctl[g, 2]:
            att += 1
            for l in range(L):
                SC_drawn[g, l] = R_drawn[g, l]
            for b in range(B):
                SC_sides[g, b] = R_sides[g, b]
                SC_owner[g, b] = R_owner[g, b]
            s1 = np.int64(R_info[g, 0])
            s2 = np.int64(R_info[g, 1])
            mover = np.int64(R_info[g, 2])
            rem = np.int64(R_info[g, 3])
            key = R_key[g]
            node = 0
            while TB[g, node, B_STATE] == ST_EXP:
                c = _puct_child(TI, TF, TS, TB, g, node, c_puct, fpu_red)
                m = np.int64(TS[g, c, S_MOVE])
                k = apply_move(SC_drawn[g], SC_sides[g], SC_owner[g], line_box, m, mover)
                if mover == 1:
                    s1 += k
                else:
                    s2 += k
                if k == 0:
                    mover = 3 - mover
                key = key ^ zob[m]
                rem -= 1
                node = c
            st = TB[g, node, B_STATE]
            if st == ST_PEND:
                break
            if st == ST_TERM or st == ST_SOLVED or st == ST_LEAF:
                _backup(TI, TF, TB, g, node, TF[g, node, F_V])
                ctl[g, 1] += 1
                continue
            diff = s1 - s2
            if mover == 2:
                diff = -diff
            rb = count_uncaptured(SC_sides[g], B)
            if rb == 0:
                v = util(diff, B)
                TB[g, node, B_STATE] = ST_TERM
                TF[g, node, F_V] = v
                _backup(TI, TF, TB, g, node, v)
                ctl[g, 1] += 1
                continue
            if node != 0:
                f = simple_value(SC_drawn[g], SC_sides[g], line_box, box_lines, B, W, cnt, memo, R)
                if f > -1000:
                    v = util(diff + f, B)
                    TB[g, node, B_STATE] = ST_SOLVED
                    TF[g, node, F_V] = v
                    _backup(TI, TF, TB, g, node, v)
                    ctl[g, 1] += 1
                    continue
            if node != 0 and rem <= leaf_solve:
                budget[0] = leaf_budget
                f = _ab(SC_drawn[g], SC_sides[g], line_box, box_lines, L, B, zob, key, rb,
                        -1000, 1000, tt_key, tt_val, tt_flag, tt_move, tt_mask, budget, 0, mbuf, F_ALL,
                        W, SEEN, cnt, memo, R)
                if budget[0] >= 0:
                    v = util(diff + f, B)
                    TB[g, node, B_STATE] = ST_SOLVED
                    TF[g, node, F_V] = v
                    _backup(TI, TF, TB, g, node, v)
                    ctl[g, 1] += 1
                    continue
            encode_state(X, nslot, SC_drawn[g], SC_sides[g], SC_owner[g], mover, diff,
                         line_box, line_cell, box_cell, L, B)
            LF_info[nslot, 0] = g
            LF_info[nslot, 1] = node
            LF_info[nslot, 2] = mover
            for l in range(L):
                LF_drawn[nslot, l] = SC_drawn[g, l]
            for b in range(B):
                LF_sides[nslot, b] = SC_sides[g, b]
            if K > 1:
                TB[g, node, B_STATE] = ST_PEND
                _vl_path(TI, g, node, vl)
            nslot += 1
            got += 1
    return nslot


@njit(cache=True)
def mcts_expand(nslot, logits, values, TI, TF, TS, TB, t_used, ctl, LF_info, LF_drawn, LF_sides,
                line_box, prm, pbuf, nbuf):
    K = np.int64(prm[4])
    vl = np.int64(prm[5])
    dir_alpha = prm[7]
    dir_eps = prm[8]
    MAXN = np.int64(prm[9])
    L = LF_drawn.shape[1]
    for s in range(nslot):
        g = LF_info[s, 0]
        node = LF_info[s, 1]
        mover = np.int64(LF_info[s, 2])
        if K > 1:
            _vl_path(TI, g, node, -vl)
        v = float(values[s])
        nl = 0
        mx = -1e30
        for l in range(L):
            if LF_drawn[s, l] == 0:
                nl += 1
                if logits[s, l] > mx:
                    mx = logits[s, l]
        if nl == 0 or t_used[g] + nl > MAXN:
            TB[g, node, B_STATE] = ST_LEAF
            TF[g, node, F_V] = v
        else:
            tot = 0.0
            for l in range(L):
                if LF_drawn[s, l] == 0:
                    e = math.exp(logits[s, l] - mx)
                    pbuf[l] = e
                    tot += e
            use_noise = node == 0 and ctl[g, 3] != 0
            ntot = 0.0
            if use_noise:
                for j in range(nl):
                    nbuf[j] = np.random.gamma(dir_alpha, 1.0)
                    ntot += nbuf[j]
                if ntot <= 0.0:
                    use_noise = False
            st = t_used[g]
            TI[g, node, I_CST] = st
            TS[g, node, S_NCH] = nl
            c = st
            j = 0
            for l in range(L):
                if LF_drawn[s, l] == 0:
                    p = pbuf[l] / tot
                    if use_noise:
                        p = (1.0 - dir_eps) * p + dir_eps * nbuf[j] / ntot
                    cap = False
                    for jj in range(2):
                        b = line_box[l, jj]
                        if b >= 0 and LF_sides[s, b] == 3:
                            cap = True
                    TS[g, c, S_MOVE] = l
                    TS[g, c, S_NCH] = 0
                    TI[g, c, I_PAR] = node
                    TI[g, c, I_CST] = 0
                    TI[g, c, I_N] = 0
                    TI[g, c, I_VL] = 0
                    TF[g, c, F_P] = p
                    TF[g, c, F_W] = 0.0
                    TF[g, c, F_V] = 0.0
                    if cap:
                        TB[g, c, B_MOVER] = mover
                    else:
                        TB[g, c, B_MOVER] = 3 - mover
                    TB[g, c, B_STATE] = ST_UNEXP
                    c += 1
                    j += 1
            t_used[g] = st + nl
            TB[g, node, B_STATE] = ST_EXP
        _backup(TI, TF, TB, g, node, v)
        ctl[g, 1] += 1


@njit(cache=True)
def root_policy(TI, TS, g, L, out):
    for l in range(L):
        out[l] = 0.0
    s = TI[g, 0, I_CST]
    n = TS[g, 0, S_NCH]
    tot = 0
    for c in range(s, s + n):
        out[TS[g, c, S_MOVE]] = TI[g, c, I_N]
        tot += TI[g, c, I_N]
    return tot


class SearchPool:
    """G parallel games, each with its own MCTS tree. Shares one solver TT."""

    def __init__(self, geo, G, max_nodes, K=1, tt_bits=22, c_puct=1.6, fpu_red=0.2,
                 leaf_solve=12, leaf_budget=5000, vl=3, dir_alpha=0.15, dir_eps=0.25,
                 max_attempts=8, solver=None):
        L, B, S = geo.L, geo.B, geo.S
        self.geo, self.G, self.K, self.MAXN = geo, G, K, int(max_nodes)
        assert self.MAXN >= L + 2
        self.TI = np.zeros((G, self.MAXN, 4), np.int32)
        self.TF = np.zeros((G, self.MAXN, 3), np.float32)
        self.TS = np.zeros((G, self.MAXN, 2), np.int16)
        self.TB = np.zeros((G, self.MAXN, 2), np.int8)
        self.t_used = np.zeros(G, np.int32)
        self.R_drawn = np.zeros((G, L), np.uint8)
        self.R_sides = np.zeros((G, B), np.int8)
        self.R_owner = np.zeros((G, B), np.int8)
        self.R_info = np.zeros((G, 4), np.int32)  # s1, s2, mover, lines left
        self.R_key = np.zeros(G, np.uint64)
        self.ctl = np.zeros((G, 4), np.int32)  # active, sims done, sims target, noise
        self.SC_drawn = np.zeros((G, L), np.uint8)
        self.SC_sides = np.zeros((G, B), np.int8)
        self.SC_owner = np.zeros((G, B), np.int8)
        P = G * K
        self.LF_info = np.zeros((P, 3), np.int32)
        self.LF_drawn = np.zeros((P, L), np.uint8)
        self.LF_sides = np.zeros((P, B), np.int8)
        self.X = np.zeros((P, C_IN, S, S), np.float32)
        self.solver = solver if solver is not None else Solver(geo, tt_bits)
        self.prm = np.array([c_puct, fpu_red, leaf_solve, leaf_budget, K, vl, max_attempts,
                             dir_alpha, dir_eps, self.MAXN], np.float64)
        self.pbuf = np.zeros(L, np.float64)
        self.nbuf = np.zeros(L, np.float64)
        self.pol = np.zeros(L, np.float32)
        self.solve_below = np.full(G, L + 1, np.int32)  # cooldown after a failed exact attempt
        for g in range(G):
            self.new_game(g)

    # ---- game state ----
    def new_game(self, g):
        self.R_drawn[g] = 0
        self.R_sides[g] = 0
        self.R_owner[g] = 0
        self.R_info[g] = (0, 0, 1, self.geo.L)
        self.R_key[g] = BASE_KEY
        self.ctl[g] = 0
        self.solve_below[g] = self.geo.L + 1

    def set_position(self, g, drawn, owner, s1, s2, mover):
        geo = self.geo
        self.R_drawn[g] = drawn
        self.R_sides[g] = drawn[geo.box_lines].sum(axis=1)
        self.R_owner[g] = owner
        self.R_info[g] = (s1, s2, mover, geo.L - int(drawn.sum()))
        self.R_key[g] = geo.key_of(drawn)
        self.ctl[g] = 0
        self.solve_below[g] = geo.L + 1

    def apply(self, g, m):
        m = int(m)
        assert self.R_drawn[g, m] == 0, "illegal move"
        info = self.R_info[g]
        mover = int(info[2])
        k = apply_move(self.R_drawn[g], self.R_sides[g], self.R_owner[g], self.geo.line_box, m, mover)
        if mover == 1:
            info[0] += k
        else:
            info[1] += k
        if k == 0:
            info[2] = 3 - mover
        info[3] -= 1
        self.R_key[g] ^= self.geo.zob[m]
        return k

    def over(self, g):
        return self.R_info[g, 3] == 0

    def remaining(self, g):
        return int(self.R_info[g, 3])

    def mover(self, g):
        return int(self.R_info[g, 2])

    def scores(self, g):
        return int(self.R_info[g, 0]), int(self.R_info[g, 1])

    def try_solve(self, g, budget):
        return self.solver.solve(self.R_drawn[g], self.R_sides[g], budget)

    def try_exact(self, g, lines_limit, budget, loony_budget, solver=None):
        """Exact solve when few lines are left OR no safe move remains (chain endgame).
        Pass a separate `solver` when calling from a worker thread."""
        rem = self.remaining(g)
        if rem >= self.solve_below[g]:
            return False, 0, []
        small = rem <= lines_limit
        if not small and count_safe_lines(self.R_drawn[g], self.R_sides[g], self.geo.line_box, self.geo.L) != 0:
            return False, 0, []
        sol = solver if solver is not None else self.solver
        ok, val, mv = sol.solve(self.R_drawn[g], self.R_sides[g], budget if small else loony_budget)
        if not ok:
            self.solve_below[g] = rem - (6 if small else 4)   # failed attempts are expensive: back off
        return ok, val, mv

    # ---- search ----
    def start_search(self, g, sims, noise=False):
        tree_reset(self.TI, self.TF, self.TS, self.TB, self.t_used, g, self.mover(g))
        self.ctl[g] = (1, 0, sims, 1 if noise else 0)

    def stop(self, g):
        self.ctl[g, 0] = 0

    def select(self):
        s, geo = self.solver, self.geo
        return mcts_select(self.TI, self.TF, self.TS, self.TB, self.t_used,
                           self.R_drawn, self.R_sides, self.R_owner, self.R_info, self.R_key, self.ctl,
                           self.SC_drawn, self.SC_sides, self.SC_owner,
                           self.LF_info, self.LF_drawn, self.LF_sides, self.X,
                           geo.line_box, geo.box_lines, geo.line_cell, geo.box_cell, geo.zob,
                           s.tt_key, s.tt_val, s.tt_flag, s.tt_move, s.tt_mask, s.budget, s.mbuf, self.prm,
                           s.W, s.SEEN, s.cnt, s.memo, LOONY_R)

    def expand(self, n, logits, values):
        mcts_expand(n, np.ascontiguousarray(logits, np.float32), np.ascontiguousarray(values, np.float32),
                    self.TI, self.TF, self.TS, self.TB, self.t_used, self.ctl,
                    self.LF_info, self.LF_drawn, self.LF_sides, self.geo.line_box,
                    self.prm, self.pbuf, self.nbuf)

    def done_games(self):
        c = self.ctl
        return np.flatnonzero((c[:, 0] == 1) & (c[:, 1] >= c[:, 2]))

    def policy(self, g):
        tot = root_policy(self.TI, self.TS, g, self.geo.L, self.pol)
        return self.pol.copy(), int(tot)

    def root_value(self, g):
        n = self.TI[g, 0, I_N]
        return float(self.TF[g, 0, F_W] / n) if n > 0 else 0.0


def util_py(margin, B):
    s = (margin > 0) - (margin < 0)
    return max(-1.0, min(1.0, 0.5 * s + 0.5 * margin / B))


# =====================================================================
# 8. REPLAY BUFFER  (numpy ring buffer, symmetries applied at sample time)
# =====================================================================

class Replay:
    FIELDS = ("drawn", "owner", "s", "mover", "pi", "full", "z", "own", "own_ok")

    def __init__(self, geo, capacity):
        L, B = geo.L, geo.B
        self.geo, self.cap = geo, int(capacity)
        self.drawn = np.zeros((self.cap, L), np.uint8)
        self.owner = np.zeros((self.cap, B), np.int8)
        self.s = np.zeros((self.cap, 2), np.int16)
        self.mover = np.zeros(self.cap, np.int8)
        self.pi = np.zeros((self.cap, L), np.float16)
        self.full = np.zeros(self.cap, np.uint8)
        self.z = np.zeros(self.cap, np.float32)
        self.own = np.zeros((self.cap, B), np.int8)
        self.own_ok = np.zeros(self.cap, np.uint8)
        self.n = self.ptr = self.total_added = 0

    def add(self, **kw):
        k = len(kw["z"])
        if k == 0:
            return
        idx = (self.ptr + np.arange(k)) % self.cap
        for f in self.FIELDS:
            getattr(self, f)[idx] = kw[f]
        self.ptr = int((self.ptr + k) % self.cap)
        self.n = min(self.cap, self.n + k)
        self.total_added += k

    def sample_idx(self, k):
        return np.random.randint(0, self.n, size=k)

    def save(self, path):
        tmp = path[:-4] + "_tmp.npz"
        np.savez_compressed(tmp, n=self.n, ptr=self.ptr, total=self.total_added,
                            **{f: getattr(self, f)[:self.n] for f in self.FIELDS})
        os.replace(tmp, path)

    def load(self, path):
        d = np.load(path)
        n = min(int(d["n"]), self.cap)
        for f in self.FIELDS:
            getattr(self, f)[:n] = d[f][-n:] if len(d[f]) > n else d[f][:n]
        self.n = n
        self.ptr = n % self.cap
        self.total_added = int(d["total"])


def make_batch(geo, replay, hard, bs, X, hard_frac=0.1):
    nh = int(bs * hard_frac) if (hard is not None and hard.n > 0) else 0
    parts = [(replay, replay.sample_idx(bs - nh))]
    if nh:
        parts.append((hard, hard.sample_idx(nh)))
    cols = {f: np.concatenate([getattr(r, f)[i] for r, i in parts]) for f in Replay.FIELDS}
    ts = np.random.randint(0, 8, size=bs)
    for t in range(1, 8):
        sel = np.flatnonzero(ts == t)
        if len(sel) == 0:
            continue
        li, bi = geo.line_inv[t], geo.box_inv[t]
        cols["drawn"][sel] = cols["drawn"][sel][:, li]
        cols["pi"][sel] = cols["pi"][sel][:, li]
        cols["owner"][sel] = cols["owner"][sel][:, bi]
        cols["own"][sel] = cols["own"][sel][:, bi]
    encode_batch(X[:bs], cols["drawn"], cols["owner"], cols["s"], cols["mover"],
                 geo.line_box, geo.box_lines, geo.line_cell, geo.box_cell, geo.L, geo.B)
    return (X[:bs], cols["pi"].astype(np.float32), cols["full"].astype(np.float32),
            cols["z"], cols["own"].astype(np.float32), cols["own_ok"].astype(np.float32),
            cols["drawn"] == 0)


# =====================================================================
# 9. SELF-PLAY  (asynchronous: every game moves as soon as its search is done)
# =====================================================================

class SelfPlay:
    """Asynchronous pipeline:
      main thread : start ready games -> select leaves (numba) -> GPU batch (own thread) -> expand
      workers     : a game whose search finished is handed to a worker thread that plays its
                    exact-solved endgame moves (own solver, GIL released). The main thread never
                    waits for the solver, so the GPU keeps getting batches while endgames are solved.
    Only the main thread touches search trees / ctl of active games; a worker only touches the
    game it owns (which is inactive while it owns it)."""

    def __init__(self, geo, cfg, evaluator, replay, threads=None):
        self.geo, self.cfg, self.ev, self.replay = geo, cfg, evaluator, replay
        G = cfg.games
        self.pool = SearchPool(geo, G, max_nodes=cfg.sims_full * geo.L + geo.L + 16, K=1,
                               tt_bits=cfg.tt_bits, c_puct=cfg.c_puct, leaf_solve=cfg.leaf_solve_lines,
                               leaf_budget=cfg.leaf_budget, dir_alpha=cfg.dir_alpha)
        self.threads = max(1, int(threads or getattr(cfg, "solver_threads", 0) or 1))
        self.solvers = queue.SimpleQueue()
        for _ in range(self.threads):
            s = Solver(geo, cfg.tt_bits)
            s.solve(np.ones(geo.L, np.uint8), np.full(geo.B, 4, np.int8), 1000)   # compile before threads start
            self.solvers.put(s)
        self.workers = ThreadPoolExecutor(self.threads)
        self.gpu = ThreadPoolExecutor(1)
        self.lock = threading.Lock()
        self.hist = [[] for _ in range(G)]
        self.moves = np.zeros(G, np.int32)
        self.full = np.zeros(G, bool)
        self.pending = []                 # search finished, not yet handed to a worker
        self.inflight = set()             # worker futures
        self.ready = queue.SimpleQueue()  # (game, full) whose endgame work is done -> start search
        self.reset_stats()
        for g in range(G):
            self._advance(g)
        self._start_ready()

    def reset_stats(self):
        self.st = dict(games=0, p1_wins=0, draws=0, abs_margin=0, positions=0, solved=0, sims=0,
                       t_select=0.0, t_gpu=0.0, t_expand=0.0, t_solver=0.0, solve_fail=0)

    def _record(self, g, pi, full):
        p = self.pool
        self.hist[g].append((p.R_drawn[g].copy(), p.R_owner[g].copy(), p.scores(g), p.mover(g),
                             pi.astype(np.float16), full))

    def _advance(self, g):
        """Worker: play exact-solved moves of game g, then mark it ready for a new search."""
        p, cfg, L = self.pool, self.cfg, self.geo.L
        sol = self.solvers.get()
        rnd = random.Random()
        t_solve = 0.0
        solved = fails = 0
        try:
            while True:
                if p.over(g):
                    self._finish(g)
                    p.new_game(g)
                    self.hist[g] = []
                    self.moves[g] = 0
                    continue
                t0 = time.time()
                before = int(p.solve_below[g])
                ok, _, mv = p.try_exact(g, cfg.root_solve_lines, cfg.root_budget, cfg.loony_budget, sol)
                t_solve += time.time() - t0
                if int(p.solve_below[g]) < before:
                    fails += 1
                if ok and mv:
                    pi = np.zeros(L, np.float32)
                    pi[mv] = 1.0 / len(mv)
                    self._record(g, pi, True)
                    p.apply(g, rnd.choice(mv))
                    self.moves[g] += 1
                    solved += 1
                    continue
                break
            self.ready.put((g, rnd.random() < cfg.p_full))
        finally:
            self.solvers.put(sol)
            with self.lock:
                self.st["solved"] += solved
                self.st["solve_fail"] += fails
                self.st["t_solver"] += t_solve

    def _start_ready(self):
        p, cfg = self.pool, self.cfg
        while True:
            try:
                g, full = self.ready.get_nowait()
            except queue.Empty:
                return
            self.full[g] = full
            p.start_search(g, cfg.sims_full if full else cfg.sims_fast, noise=full)

    def _pick(self, g):
        """Search of game g finished: record its policy target and play the move."""
        p, cfg = self.pool, self.cfg
        visits, tot = p.policy(g)
        pi = visits / max(tot, 1)
        full = bool(self.full[g])
        self._record(g, pi, full)
        if full and self.moves[g] < cfg.temp_moves and tot > 0:
            w = pi.astype(np.float64) ** (1.0 / cfg.temperature)
            m = int(np.random.choice(self.geo.L, p=w / w.sum()))
        else:
            m = int(random.choice(np.flatnonzero(visits == visits.max())))
        p.stop(g)
        p.apply(g, m)
        self.moves[g] += 1

    def _on_done(self, g):           # synchronous helper (tests / tools)
        self._pick(g)
        self._advance(g)
        self._start_ready()

    def _finish(self, g):
        h = self.hist[g]
        if not h:
            return
        geo, p = self.geo, self.pool
        s1, s2 = p.scores(g)
        fown = p.R_owner[g].copy()
        movers = np.array([x[3] for x in h], np.int8)
        margin = np.where(movers == 1, s1 - s2, s2 - s1)
        data = dict(
            drawn=np.stack([x[0] for x in h]), owner=np.stack([x[1] for x in h]),
            s=np.array([x[2] for x in h], np.int16), mover=movers,
            pi=np.stack([x[4] for x in h]), full=np.array([x[5] for x in h], np.uint8),
            z=np.array([util_py(int(mg), geo.B) for mg in margin], np.float32),
            own=np.where(fown[None, :] == movers[:, None], 1, -1).astype(np.int8),
            own_ok=np.ones(len(h), np.uint8))
        with self.lock:
            self.replay.add(**data)
            st = self.st
            st["games"] += 1
            st["p1_wins"] += int(s1 > s2)
            st["draws"] += int(s1 == s2)
            st["abs_margin"] += abs(s1 - s2)
            st["positions"] += len(h)

    def _reap(self, block=False):
        if not self.inflight:
            return
        done, _ = wait(self.inflight, timeout=(0.05 if block else 0), return_when=FIRST_COMPLETED)
        for f in done:
            self.inflight.discard(f)
            f.result()                       # re-raise worker errors in the main thread

    def step(self):
        """One pipelined round. Returns the number of leaves evaluated."""
        pool, st = self.pool, self.st
        self._start_ready()
        t0 = time.time()
        n = pool.select()
        t1 = time.time()
        fut = self.gpu.submit(self.ev, pool.X[:n]) if n > 0 else None
        for g in self.pending:              # games in `pending` are stopped -> not in this batch
            self.inflight.add(self.workers.submit(self._advance, g))
        self.pending = []
        if fut is not None:
            logits, values = fut.result()
            t2 = time.time()
            pool.expand(n, logits, values)
            t3 = time.time()
            st["sims"] += n
            st["t_gpu"] += t2 - t1
            st["t_expand"] += t3 - t2
        st["t_select"] += t1 - t0
        done = pool.done_games()
        for g in done:
            self._pick(g)
        self.pending = list(done)
        self._reap(block=(n == 0 and not self.pending))
        return n

    def drain(self):
        """Wait for every worker, so the replay buffer is quiet (call before training)."""
        for g in self.pending:
            self.inflight.add(self.workers.submit(self._advance, g))
        self.pending = []
        for f in list(self.inflight):
            f.result()
        self.inflight.clear()

    def run(self, min_positions):
        target = self.replay.total_added + min_positions
        while self.replay.total_added < target:
            self.step()
        self.drain()

    def close(self):
        try:
            self.drain()
        finally:
            self.workers.shutdown()
            self.gpu.shutdown()


# =====================================================================
# 10. TRAINING STEP
# =====================================================================

def train_step(model, opt, scaler, ev, batch, cfg):
    X, pi, full, z, own, own_ok, legal = batch
    dev = ev.device
    x = torch.from_numpy(X).to(dev, non_blocking=True)
    if ev.cl:
        x = x.contiguous(memory_format=torch.channels_last)
    pi_t = torch.from_numpy(pi).to(dev)
    full_t = torch.from_numpy(full).to(dev)
    z_t = torch.from_numpy(z).to(dev)
    own_t = torch.from_numpy(own).to(dev)
    ok_t = torch.from_numpy(own_ok).to(dev)
    legal_t = torch.from_numpy(legal).to(dev)
    with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=ev.amp):
        logits, v, o = model(x, ev.line_flat, ev.box_flat)
    logits = logits.float().masked_fill(~legal_t, -1e4)
    logp = F.log_softmax(logits, dim=1)
    pl = (-(pi_t * logp).sum(1) * full_t).sum() / full_t.sum().clamp(min=1.0)
    vl = F.mse_loss(v.float(), z_t)
    ol = (((o.float() - own_t) ** 2).mean(1) * ok_t).sum() / ok_t.sum().clamp(min=1.0)
    loss = pl + cfg.w_value * vl + cfg.w_own * ol
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    return float(pl.detach()), float(vl.detach()), float(ol.detach())


# =====================================================================
# 11. ARENA  (evaluation: net vs heuristic bot, new net vs best net)
# =====================================================================

def random_safe_opening(geo, n_moves, rng):
    drawn = np.zeros(geo.L, np.uint8)
    sides = np.zeros(geo.B, np.int8)
    seq = []
    for _ in range(n_moves):
        safe = [l for l in range(geo.L)
                if not drawn[l] and all(b < 0 or sides[b] < 2 for b in geo.line_box[l])]
        if not safe:
            break
        l = rng.choice(safe)
        seq.append(l)
        drawn[l] = 1
        for b in geo.line_box[l]:
            if b >= 0:
                sides[b] += 1
    return seq


def arena(geo, cfg, agents, n_games, sims, seed=0):
    """agents[i] = {'kind': 'net', 'eval': NetEvaluator} or {'kind': 'bot', 'solve_lines': int}.
    Agent 0 is player 1 in even games and player 2 in odd games (same random opening per pair).
    Returns results from agent 0's point of view."""
    n_games += n_games % 2
    rng = random.Random(seed)
    pool = SearchPool(geo, n_games, max_nodes=sims * geo.L + geo.L + 16, K=1, tt_bits=cfg.tt_bits,
                      c_puct=cfg.c_puct, leaf_solve=cfg.leaf_solve_lines, leaf_budget=cfg.leaf_budget)
    side0 = np.array([1 if g % 2 == 0 else 2 for g in range(n_games)])
    agent_of = np.zeros(n_games, np.int8)
    finished = np.zeros(n_games, bool)
    stack = np.zeros(geo.L + 4, np.int64)
    cand = np.zeros(geo.L, np.int64)
    for g in range(0, n_games, 2):
        for m in random_safe_opening(geo, max(2, geo.L // 10), rng):
            pool.apply(g, m)
            pool.apply(g + 1, m)

    def advance(g):
        while not pool.over(g):
            a = 0 if pool.mover(g) == side0[g] else 1
            ag = agents[a]
            if ag["kind"] == "bot":
                ok, _, mv = pool.try_exact(g, ag.get("solve_lines", 0), cfg.root_budget, cfg.loony_budget)
                if ok:
                    pool.apply(g, rng.choice(mv))
                    continue
                pool.apply(g, int(heuristic_move(pool.R_drawn[g], pool.R_sides[g], geo.line_box,
                                                 geo.box_lines, geo.L, geo.B, stack, cand)))
                continue
            ok, _, mv = pool.try_exact(g, cfg.root_solve_lines, cfg.root_budget, cfg.loony_budget)
            if ok:
                pool.apply(g, rng.choice(mv))
                continue
            agent_of[g] = a
            pool.start_search(g, sims, noise=False)
            return
        finished[g] = True

    for g in range(n_games):
        advance(g)
    while not finished.all():
        n = pool.select()
        if n > 0:
            ag = agent_of[pool.LF_info[:n, 0]]
            logits = np.zeros((n, geo.L), np.float32)
            values = np.zeros(n, np.float32)
            for a in (0, 1):
                sel = np.flatnonzero(ag == a)
                if len(sel):
                    lg, vv = agents[a]["eval"](pool.X[sel])
                    logits[sel], values[sel] = lg, vv
            pool.expand(n, logits, values)
        for g in pool.done_games():
            visits, _ = pool.policy(g)
            pool.stop(g)
            pool.apply(g, int(rng.choice(list(np.flatnonzero(visits == visits.max())))))
            advance(g)
    w = d = l = 0
    margin = 0
    for g in range(n_games):
        s1, s2 = pool.scores(g)
        mg = (s1 - s2) if side0[g] == 1 else (s2 - s1)
        margin += mg
        w += mg > 0
        d += mg == 0
        l += mg < 0
    return dict(games=n_games, wins=w, draws=d, losses=l, score=(w + 0.5 * d) / n_games,
                avg_margin=margin / n_games)


# =====================================================================
# 12. TRAINING LOOP  (time budget, crash-safe checkpoints, resume, arena)
# =====================================================================

def board_defaults(args, geo):
    N, L = geo.N, geo.L
    table = {6: (400, 80, 32, 14), 5: (320, 64, 24, 14)}   # exact-solve threshold ~40% of the lines
    full, fast, root_lines, leaf_lines = table.get(N, (96, 24, 14, 10))
    for name, val in (("sims_full", full), ("sims_fast", fast), ("root_solve_lines", root_lines),
                      ("leaf_solve_lines", leaf_lines), ("temp_moves", L // 4),
                      ("dir_alpha", round(10.0 / L, 3)), ("replay", 60000 * N)):
        if getattr(args, name, None) is None:
            setattr(args, name, val)
    return args


def pick_device(name):
    if torch is None:
        return None
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    nb_seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def atomic_torch_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


class Logger:
    def __init__(self, path):
        self.path = path

    def __call__(self, msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def load_net(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = ck.get("net", {"channels": 64, "blocks": 8})
    model = DABNet(net["channels"], net["blocks"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def train(args):
    if torch is None:
        sys.exit("PyTorch is required for training")
    geo = geometry(args.board)
    board_defaults(args, geo)
    seed_everything(args.seed)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    out = args.out
    os.makedirs(out, exist_ok=True)
    latest_path = os.path.join(out, "latest.pt")
    if args.resume_from and not os.path.exists(latest_path):
        for fname in ("latest.pt", "best.pt", "replay.npz", "metrics.csv", "train_log.txt"):
            src = os.path.join(args.resume_from, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out, fname))
    log = Logger(os.path.join(out, "train_log.txt"))
    net_cfg = {"channels": args.channels, "blocks": args.blocks}

    model = DABNet(args.channels, args.blocks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    state = dict(iters=0, games=0, positions=0, steps=0, hours=0.0, promotions=0)
    if os.path.exists(latest_path):
        ck = torch.load(latest_path, map_location=device, weights_only=False)
        if ck.get("net") != net_cfg:
            sys.exit(f"checkpoint net {ck.get('net')} != requested {net_cfg}")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        state.update(ck["state"])
        log(f"RESUMED {latest_path}: {state['games']:,} games, {state['hours']:.2f} h trained so far")
    elif args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location=device, weights_only=False)["model"])
        log(f"initialised weights from {args.init_from}")
    for pg in opt.param_groups:
        pg["lr"] = args.lr
    model.eval()

    replay = Replay(geo, args.replay)
    replay_path = os.path.join(out, "replay.npz")
    if os.path.exists(replay_path):
        replay.load(replay_path)
        log(f"replay loaded: {replay.n:,} positions")
    hard = None
    if args.hard_file and os.path.exists(args.hard_file):
        hard = Replay(geo, 200_000)
        hard.load(args.hard_file)
        log(f"hard positions (from human games) loaded: {hard.n:,}")

    best = DABNet(args.channels, args.blocks).to(device)
    best_path = os.path.join(out, "best.pt")
    if os.path.exists(best_path):
        best.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model"])
    else:
        best.load_state_dict(model.state_dict())
        atomic_torch_save({"model": best.state_dict(), "net": net_cfg, "board": geo.N, "state": dict(state)},
                          best_path)
    best.eval()

    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
        best.to(memory_format=torch.channels_last)
    ev = NetEvaluator(model, geo, device)
    ev_best = NetEvaluator(best, geo, device)
    threads = args.solver_threads or max(1, (os.cpu_count() or 2) // max(1, args.procs_sharing_cpu))
    args.solver_threads = threads
    log(f"board {geo.N}x{geo.N} | device {device} | cpus {os.cpu_count()} solver threads {threads} | "
        f"games in parallel {args.games} | sims {args.sims_full}/{args.sims_fast} | exact solver <= "
        f"{args.root_solve_lines} lines or chain endgame (+chain theory) | net {args.channels}x{args.blocks} | "
        f"session limit {args.hours:.2f} h")
    sp = SelfPlay(geo, args, ev, replay, threads)
    Xb = np.zeros((args.batch, C_IN, geo.S, geo.S), np.float32)

    t_start = time.time()
    hours_before = state["hours"]
    deadline = t_start + args.hours * 3600.0
    last_save = time.time()
    next_eval = t_start + args.first_eval_minutes * 60
    iter_time = 0.0
    metrics_path = os.path.join(out, "metrics.csv")
    status_path = os.path.join(out, "status.json")
    stop_flag = {"stop": False}
    try:
        import signal
        signal.signal(signal.SIGTERM, lambda *a: stop_flag.update(stop=True))
    except Exception:
        pass
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w") as f:
            f.write("hours,iter,games,positions,sims_per_s,policy_loss,value_loss,own_loss,p1_win,draw,"
                    "abs_margin,solved_frac,replay,vs_bot,vs_best,lr,vs_bot_margin\n")
    last_arena = {}

    def save_all(tag):
        state["hours"] = hours_before + (time.time() - t_start) / 3600.0
        atomic_torch_save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                           "state": dict(state), "net": net_cfg, "board": geo.N, "version": 3}, latest_path)
        replay.save(replay_path)
        log(f"checkpoint saved ({tag})")

    def lr_at(elapsed_frac, steps_done):
        warm = min(1.0, (steps_done + 1) / 300.0)
        f = min(1.0, max(0.0, elapsed_frac))
        return args.lr * warm * (args.lr_floor + (1 - args.lr_floor) * 0.5 * (1 + math.cos(math.pi * f)))

    def run_arena(tag):
        r1 = arena(geo, args, [{"kind": "net", "eval": ev}, {"kind": "bot", "solve_lines": args.root_solve_lines}],
                   args.eval_games, args.eval_sims, seed=state["iters"])
        r2 = arena(geo, args, [{"kind": "net", "eval": ev}, {"kind": "net", "eval": ev_best}],
                   args.eval_games, args.eval_sims, seed=state["iters"] + 7)
        msg = (f"ARENA[{tag}] vs heuristic+solver bot: W{r1['wins']} D{r1['draws']} L{r1['losses']} "
               f"avg margin {r1['avg_margin']:+.1f} | vs best: W{r2['wins']} D{r2['draws']} L{r2['losses']} "
               f"score {r2['score']:.3f}")
        if r2["score"] >= 0.55:
            best.load_state_dict(model.state_dict())
            state["promotions"] += 1
            atomic_torch_save({"model": best.state_dict(), "net": net_cfg, "board": geo.N,
                               "state": dict(state), "vs_bot": r1}, best_path)
            msg += "  -> NEW BEST saved"
        log(msg)
        last_arena.update(vs_bot=r1["score"], vs_bot_margin=r1["avg_margin"], vs_best=r2["score"],
                          at_hours=round(hours_before + (time.time() - t_start) / 3600.0, 3))
        return r1, r2

    if state["iters"] == 0 and args.first_eval_minutes > 0:
        r1 = arena(geo, args, [{"kind": "net", "eval": ev}, {"kind": "bot", "solve_lines": args.root_solve_lines}],
                   args.eval_games, args.eval_sims, seed=12345)
        log(f"BASELINE (untrained net) vs heuristic+solver bot: W{r1['wins']} D{r1['draws']} L{r1['losses']} "
            f"avg margin {r1['avg_margin']:+.1f}")

    crashed = None
    try:
        while True:
            now = time.time()
            if args.max_iters and state["iters"] >= args.max_iters:
                break
            if now + 1.3 * iter_time + args.reserve_minutes * 60 > deadline:
                break
            if stop_flag["stop"] or os.path.exists(os.path.join(out, "STOP")):
                log("stop requested")
                break
            t0 = time.time()
            sp.reset_stats()
            model.eval()
            sp.run(args.positions_per_iter)
            t_sp = time.time() - t0
            st = dict(sp.st)
            losses = []
            lr = lr_at((time.time() - t_start) / (args.hours * 3600.0), state["steps"])
            if replay.n >= args.min_replay:
                steps = max(1, int(st["positions"] * args.train_ratio / args.batch))
                model.train()
                for _ in range(steps):
                    lr = lr_at((time.time() - t_start) / (args.hours * 3600.0), state["steps"])
                    for pg in opt.param_groups:
                        pg["lr"] = lr
                    losses.append(train_step(model, opt, scaler, ev,
                                             make_batch(geo, replay, hard, args.batch, Xb), args))
                    state["steps"] += 1
                model.eval()
            state["iters"] += 1
            state["games"] += st["games"]
            state["positions"] += st["positions"]
            lp = np.mean(losses, axis=0) if losses else (float("nan"),) * 3
            ng = max(1, st["games"])
            vs_bot = vs_best = vs_margin = ""
            if args.eval_minutes > 0 and time.time() >= next_eval and replay.n >= args.min_replay:
                r1, r2 = run_arena(f"it{state['iters']}")
                vs_bot, vs_best, vs_margin = f"{r1['score']:.3f}", f"{r2['score']:.3f}", f"{r1['avg_margin']:.2f}"
                next_eval = time.time() + args.eval_minutes * 60
            iter_time = time.time() - t0
            elapsed_h = hours_before + (time.time() - t_start) / 3600.0
            sps = st["sims"] / max(t_sp, 1e-6)
            log(f"it {state['iters']:4d} | {elapsed_h:5.2f}h | games {state['games']:,} (+{st['games']}) | "
                f"sims/s {sps:,.0f} | loss p {lp[0]:.3f} v {lp[1]:.3f} own {lp[2]:.3f} | lr {lr:.2e} | "
                f"P1 win {st['p1_wins'] / ng:.0%} draw {st['draws'] / ng:.0%} | |margin| {st['abs_margin'] / ng:.1f} | "
                f"exact moves {st['solved'] / max(1, st['positions']):.0%} | replay {replay.n:,} | "
                f"{iter_time:.0f}s/it | time: select {st['t_select']:.0f}s gpu {st['t_gpu']:.0f}s "
                f"expand {st['t_expand']:.0f}s solver {st['t_solver']:.0f}s-thr ({st['solve_fail']} failed solves)")
            with open(metrics_path, "a") as f:
                f.write(f"{elapsed_h:.3f},{state['iters']},{state['games']},{state['positions']},"
                        f"{sps:.0f},{lp[0]:.4f},{lp[1]:.4f},{lp[2]:.4f},"
                        f"{st['p1_wins'] / ng:.3f},{st['draws'] / ng:.3f},{st['abs_margin'] / ng:.2f},"
                        f"{st['solved'] / max(1, st['positions']):.3f},{replay.n},{vs_bot},{vs_best},{lr:.3e},"
                        f"{vs_margin}\n")
            try:
                with open(status_path + ".tmp", "w") as f:
                    json.dump(dict(board=geo.N, hours=round(elapsed_h, 3),
                                   session_left_h=round((deadline - time.time()) / 3600, 3),
                                   iters=state["iters"], games=state["games"], sims_per_s=round(sps),
                                   policy_loss=float(lp[0]), value_loss=float(lp[1]), lr=lr,
                                   promotions=state["promotions"], last_arena=last_arena,
                                   time=time.strftime("%Y-%m-%d %H:%M:%S")), f)
                os.replace(status_path + ".tmp", status_path)
            except Exception:
                pass
            if time.time() - last_save > args.save_minutes * 60:
                save_all("periodic")
                last_save = time.time()
        if args.eval_minutes > 0 and replay.n >= args.min_replay and time.time() + 120 < deadline:
            run_arena("final")
    except BaseException as e:            # save whatever we have, then re-raise
        crashed = e
        log(f"ERROR: {type(e).__name__}: {e} -> saving checkpoint before exit")
    finally:
        sp.close()
        save_all("final" if crashed is None else "crash")
        atomic_torch_save({"model": model.state_dict(), "net": net_cfg, "board": geo.N, "state": dict(state)},
                          os.path.join(out, "final.pt"))
    if crashed is not None:
        raise crashed
    log("training session finished; outputs in " + out)


# =====================================================================
# 13. LOSS BOOK  (AI remembers corrections from games it lost or drew)
# =====================================================================
#  Keys are canonical over the 8 board symmetries, so a mirrored or rotated
#  version of the same trick is also caught.

class LossBook:
    def __init__(self, path):
        self.path = path
        self.data = {}
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                self.data = pickle.load(f)

    def lookup(self, geo, drawn):
        key, t = geo.canonical(drawn)
        cm = self.data.get((geo.N, key))
        if cm is None:
            return None
        m = int(geo.line_inv[t][cm])
        return m if drawn[m] == 0 else None

    def store(self, geo, drawn, move):
        key, t = geo.canonical(drawn)
        self.data[(geo.N, key)] = int(geo.line_perm[t][move])

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.path)


# =====================================================================
# 14. AI MOVE SELECTION FOR PLAY  (book -> exact solver -> deep MCTS)
# =====================================================================

def ai_think(pool, ev, geo, think_s, solve_nodes, book=None, solve_lines=40, min_sims=64):
    drawn = pool.R_drawn[0]
    if book is not None:
        m = book.lookup(geo, drawn)
        if m is not None:
            return m, {"src": "book"}
    if pool.remaining(0) <= solve_lines or count_safe_lines(drawn, pool.R_sides[0], geo.line_box, geo.L) == 0:
        t = time.time()
        ok, val, mv = pool.try_solve(0, solve_nodes)
        if ok:
            pi = np.zeros(geo.L, np.float32)
            pi[mv] = 1.0 / len(mv)
            s1, s2 = pool.scores(0)
            d = (s1 - s2) if pool.mover(0) == 1 else (s2 - s1)
            return random.choice(mv), {"src": "exact", "value": val, "q": util_py(d + val, geo.B),
                                       "pi": pi, "secs": time.time() - t}
    pool.start_search(0, 1 << 30, noise=False)
    t0 = time.time()
    guard = pool.MAXN - geo.L * pool.K - 8
    while True:
        n = pool.select()
        if n:
            lg, vv = ev(pool.X[:n])
            pool.expand(n, lg, vv)
        el = time.time() - t0
        if el >= think_s and (pool.TI[0, 0, I_N] >= min_sims or n == 0):
            break
        if pool.t_used[0] >= guard:
            break
    visits, tot = pool.policy(0)
    q = pool.root_value(0)
    pool.stop(0)
    m = int(np.argmax(visits))
    return m, {"src": "mcts", "visits": tot, "q": q, "pi": visits / max(tot, 1), "secs": time.time() - t0}


# =====================================================================
# 15. TERMINAL PLAY
# =====================================================================

HELP_TEXT = """
Moves use DOT coordinates (row col), dots are the 'o' marks:
   h r c      horizontal line from dot (r,c) to (r,c+1)      e.g.  h 0 2
   v r c      vertical   line from dot (r,c) to (r+1,c)      e.g.  v 1 3
   r1 c1 r2 c2  the two dots directly                        e.g.  0 2 0 3
Commands: hint, help, quit.   Y = your box, A = AI box, === / # = AI's last moves
"""


def render(geo, drawn, owner, human, highlight=()):
    N = geo.N
    hl = set(int(x) for x in highlight)
    rows = ["     " + "".join(f"{c:<4}" for c in range(N + 1))]
    for r in range(N + 1):
        row = f"{r:>3}  "
        for c in range(N + 1):
            row += "o"
            if c < N:
                l = geo.h(r, c)
                row += ("===" if l in hl else "---") if drawn[l] else "   "
        rows.append(row)
        if r < N:
            row = "     "
            for c in range(N + 1):
                l = geo.v(r, c)
                row += ("#" if l in hl else "|") if drawn[l] else " "
                if c < N:
                    o = owner[r * N + c]
                    row += "   " if o == 0 else (" Y " if o == human else " A ")
            rows.append(row)
    return "\n".join(rows)


def parse_move(geo, text):
    t = text.replace(",", " ").split()
    N = geo.N
    try:
        if len(t) == 3 and t[0] in ("h", "v"):
            r, c = int(t[1]), int(t[2])
            if t[0] == "h" and 0 <= r <= N and 0 <= c < N:
                return geo.h(r, c)
            if t[0] == "v" and 0 <= r < N and 0 <= c <= N:
                return geo.v(r, c)
        if len(t) == 4:
            r1, c1, r2, c2 = map(int, t)
            if r1 == r2 and abs(c1 - c2) == 1 and 0 <= r1 <= N and 0 <= min(c1, c2) < N:
                return geo.h(r1, min(c1, c2))
            if c1 == c2 and abs(r1 - r2) == 1 and 0 <= c1 <= N and 0 <= min(r1, r2) < N:
                return geo.v(min(r1, r2), c1)
    except ValueError:
        pass
    return None


def ask(prompt):
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return "quit"


def make_play_pool(geo, args):
    return SearchPool(geo, 1, max_nodes=args.max_nodes, K=args.leaves, tt_bits=args.tt_bits,
                      leaf_solve=args.leaf_solve_lines, leaf_budget=args.leaf_budget)


def append_hard_positions(path, geo, samples):
    rep = Replay(geo, 200_000)
    if os.path.exists(path):
        rep.load(path)
    k = len(samples)
    rep.add(drawn=np.stack([x[0] for x in samples]), owner=np.stack([x[1] for x in samples]),
            s=np.array([x[2] for x in samples], np.int16), mover=np.array([x[3] for x in samples], np.int8),
            pi=np.stack([x[4] for x in samples]).astype(np.float16), full=np.ones(k, np.uint8),
            z=np.array([x[5] for x in samples], np.float32), own=np.zeros((k, geo.B), np.int8),
            own_ok=np.zeros(k, np.uint8))
    rep.save(path)


def learn_from_game(rec, geo, ev, args, book, hard_path, progress=None, log=print):
    """Re-analyse every AI decision of a game with a much deeper search.
    Different move -> stored in the loss book, so the very same position never gets the
    same reply twice: one lost game is enough to kill that line for good.
    All analysed positions -> hard-position file (use with train --hard-file)."""
    pool = make_play_pool(geo, args)
    human = rec["human"]
    ai_i = changed = 0
    samples = []
    n_ai = sum(1 for s in rec.get("ai", []) if s != "exact") or 1
    for idx, m in enumerate(rec["moves"]):
        if pool.mover(0) != human:
            src = rec["ai"][ai_i] if ai_i < len(rec["ai"]) else "mcts"
            ai_i += 1
            if src != "exact":
                if progress is not None:
                    progress(ai_i, n_ai, changed)
                best, info = ai_think(pool, ev, geo, args.learn_think, args.solve_nodes * 4, None,
                                      solve_lines=args.solve_lines + 6)
                if "pi" in info:
                    samples.append((pool.R_drawn[0].copy(), pool.R_owner[0].copy(), pool.scores(0),
                                    pool.mover(0), info["pi"], info["q"]))
                if best != m:
                    book.store(geo, pool.R_drawn[0], best)
                    changed += 1
                    log(f"  move {idx + 1}: AI played {geo.describe_line(m)}, deeper analysis prefers "
                        f"{geo.describe_line(best)} ({info['src']})")
        pool.apply(0, m)
    book.save()
    if samples:
        append_hard_positions(hard_path, geo, samples)
    log(f"learned: {changed} corrections in loss book, {len(samples)} positions saved to {hard_path}")
    return changed


def play(args):
    if torch is None:
        sys.exit("PyTorch is required to play")
    geo = geometry(args.board)
    device = pick_device(args.device)
    model, ck = load_net(args.model, device)
    if ck.get("board") not in (None, geo.N):
        print(f"note: model trained on {ck['board']}x{ck['board']}, you are playing {geo.N}x{geo.N}")
    ev = NetEvaluator(model, geo, device)
    os.makedirs(args.data_dir, exist_ok=True)
    book = LossBook(os.path.join(args.data_dir, f"book_{geo.N}.pkl"))
    hard_path = os.path.join(args.data_dir, f"hard_{geo.N}.npz")
    pool = make_play_pool(geo, args)
    human = {"human": 1, "ai": 2}.get(args.first, random.choice([1, 2]))
    rec = {"board": geo.N, "human": human, "moves": [], "ai": []}
    print(HELP_TEXT)
    print(f"{geo.N}x{geo.N} boxes | you are {'first' if human == 1 else 'second'} | AI thinks {args.think:.0f}s per move"
          f" | loss book entries: {len(book.data)}")
    highlight = []
    last_mover = None
    while not pool.over(0):
        s1, s2 = pool.scores(0)
        you, ai = (s1, s2) if human == 1 else (s2, s1)
        print()
        print(render(geo, pool.R_drawn[0], pool.R_owner[0], human, highlight))
        print(f"You (Y): {you}   AI (A): {ai}   lines left: {pool.remaining(0)}")
        if pool.mover(0) == human:
            text = ask("your move> ")
            if text in ("q", "quit", "exit"):
                print("game abandoned")
                return
            if text in ("help", "?"):
                print(HELP_TEXT)
                continue
            if text == "hint":
                m, _ = ai_think(pool, ev, geo, args.think, args.solve_nodes, None, solve_lines=args.solve_lines)
                print("hint:", geo.describe_line(m))
                continue
            m = parse_move(geo, text)
            if m is None or pool.R_drawn[0, m]:
                print("invalid or already drawn. Type 'help' for the move format.")
                continue
            k = pool.apply(0, m)
            rec["moves"].append(int(m))
            if k:
                print(f"you completed {k} box(es) - move again")
            last_mover = "human"
        else:
            m, info = ai_think(pool, ev, geo, args.think, args.solve_nodes, book, solve_lines=args.solve_lines)
            if last_mover != "ai":
                highlight = []
            k = pool.apply(0, m)
            rec["moves"].append(int(m))
            rec["ai"].append(info["src"])
            highlight.append(m)
            last_mover = "ai"
            if info["src"] == "exact":
                detail = f"exact solve, AI nets {info['value']:+d} of the remaining boxes"
            elif info["src"] == "book":
                detail = "loss book"
            else:
                detail = f"{info['visits']:,} visits in {info['secs']:.1f}s, eval {info['q']:+.2f}"
            print(f"AI plays {geo.describe_line(m)}   [{detail}]" + ("  +box, moves again" if k else ""))
    s1, s2 = pool.scores(0)
    you, ai = (s1, s2) if human == 1 else (s2, s1)
    print()
    print(render(geo, pool.R_drawn[0], pool.R_owner[0], human, highlight))
    print(f"FINAL  You {you} - {ai} AI  ->  " + ("YOU WIN" if you > ai else "DRAW" if you == ai else "AI WINS"))
    rec.update(you=you, ai_score=ai, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(args.data_dir, f"games_{geo.N}.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    if you >= ai - args.learn_margin and not args.no_learn:
        print("Re-analysing this game so the same idea never works twice (Ctrl-C to skip)...")
        try:
            learn_from_game(rec, geo, ev, args, book, hard_path)
        except KeyboardInterrupt:
            print("analysis skipped")
        else:
            learned_path = os.path.join(args.data_dir, f"learned_{geo.N}.json")
            done = json.load(open(learned_path)) if os.path.exists(learned_path) else []
            json.dump(done + [rec["time"]], open(learned_path, "w"))


def learn_cmd(args):
    """Re-analyse all saved games the AI lost or drew (can run on Kaggle GPU with long think time)."""
    geo = geometry(args.board)
    device = pick_device(args.device)
    model, _ = load_net(args.model, device)
    ev = NetEvaluator(model, geo, device)
    games_path = os.path.join(args.data_dir, f"games_{geo.N}.jsonl")
    if not os.path.exists(games_path):
        sys.exit(f"no saved games at {games_path}")
    book = LossBook(os.path.join(args.data_dir, f"book_{geo.N}.pkl"))
    hard_path = os.path.join(args.data_dir, f"hard_{geo.N}.npz")
    learned_path = os.path.join(args.data_dir, f"learned_{geo.N}.json")
    done = set(json.load(open(learned_path))) if os.path.exists(learned_path) else set()
    recs = [json.loads(x) for x in open(games_path) if x.strip()]
    todo = [r for r in recs if r["you"] >= r["ai_score"] and r["time"] not in done]
    print(f"{len(todo)} games to analyse")
    for r in todo:
        print(f"game {r['time']}: you {r['you']} - {r['ai_score']} AI")
        learn_from_game(r, geo, ev, args, book, hard_path)
        done.add(r["time"])
        json.dump(sorted(done), open(learned_path, "w"))


# =====================================================================
# 16. TESTS AND BENCHMARK
# =====================================================================

def _brute(geo, drawn, sides, memo):
    key = drawn.tobytes()
    if key in memo:
        return memo[key]
    if (sides == 4).all():
        return 0
    best = -1000
    for l in range(geo.L):
        if drawn[l]:
            continue
        drawn[l] = 1
        k = 0
        for b in geo.line_box[l]:
            if b >= 0:
                sides[b] += 1
                k += sides[b] == 4
        v = k + _brute(geo, drawn, sides, memo) if k else -_brute(geo, drawn, sides, memo)
        drawn[l] = 0
        for b in geo.line_box[l]:
            if b >= 0:
                sides[b] -= 1
        best = max(best, v)
    memo[key] = best
    return best


def run_tests(quick=False):
    t0 = time.time()
    rng = random.Random(11)
    seed_everything(11)

    def ok(msg):
        print(f"[{time.time() - t0:5.1f}s] PASS  {msg}", flush=True)

    for N in (2, 3, 5, 6):
        g = geometry(N)
        for t in range(8):
            assert sorted(g.line_perm[t]) == list(range(g.L))
            for b in range(g.B):
                assert set(g.line_perm[t][g.box_lines[b]]) == set(g.box_lines[g.box_perm[t][b]])
    ok("8 symmetries preserve box structure (2x2, 3x3, 5x5, 6x6)")

    for N in (5, 6):
        g = geometry(N)
        pool = SearchPool(g, 1, max_nodes=g.L + 4)
        for _ in range(20):
            pool.new_game(0)
            while not pool.over(0):
                pool.apply(0, rng.choice(list(np.flatnonzero(pool.R_drawn[0] == 0))))
            s1, s2 = pool.scores(0)
            assert s1 + s2 == g.B and (pool.R_sides[0] == 4).all()
    ok("random games: every box scored exactly once (5x5, 6x6)")

    for N, rem, trials in ((2, 8, 80), (3, 11, 40), (3, 13, 15)):
        g = geometry(N)
        for _ in range(trials):
            pool = SearchPool(g, 1, max_nodes=g.L + 4)
            order = list(range(g.L))
            rng.shuffle(order)
            for m in order[:g.L - rem]:
                pool.apply(0, m)
            d, s = pool.R_drawn[0].copy(), pool.R_sides[0].copy()
            vals, memo = {}, {}
            for l in np.flatnonzero(d == 0):
                d2, s2 = d.copy(), s.copy()
                d2[l] = 1
                k = 0
                for b in g.line_box[l]:
                    if b >= 0:
                        s2[b] += 1
                        k += s2[b] == 4
                vals[int(l)] = k + _brute(g, d2, s2, memo) if k else -_brute(g, d2, s2, memo)
            bv = max(vals.values())
            best_set = sorted(m for m in vals if vals[m] == bv)
            for prune in (F_ALL, 0, F_PRUNE, F_PRUNE | F_DEDUP, F_PRUNE | F_THEORY, F_THEORY, F_DEDUP):
                good, v, mv = Solver(g, 16).solve(d, s, 10 ** 9, prune)
                assert good and v == bv and all(vals[m] == bv for m in mv), (N, rem, prune, v, bv)
                if prune & F_PRUNE == 0:
                    assert sorted(mv) == best_set, (N, rem, prune, mv, best_set)
            if quick and _ >= 5:
                break
    ok("exact solver == brute-force minimax (2x2, 3x3; every speed-up flag combination)")

    # chain theory + de-duplication on realistic big-board endgames vs the plain pruned search
    n_simple = n_cmp = 0
    cases = ((4, 22, 10), (5, 24, 8)) if quick else ((4, 22, 40), (5, 24, 40), (5, 28, 25), (6, 26, 25))
    for N, rem, trials in cases:
        g = geometry(N)
        ref, fast = Solver(g, 20), Solver(g, 20)
        for _ in range(trials):
            d, s = bot_position(g, rem, rng)
            ok1, v1, m1 = ref.solve(d, s, 30_000_000, F_PRUNE)
            ok2, v2, m2 = fast.solve(d, s, 30_000_000, F_ALL)
            if not ok1:
                continue
            assert ok2 and v1 == v2 and sorted(m1) == sorted(m2), (N, rem, v1, v2, m1, m2)
            n_cmp += 1
    # pure simple-loony positions (only independent chains and loops) vs the plain search
    for N in ((4,) if quick else (3, 4, 5)):
        g = geometry(N)
        ref, fast = Solver(g, 20), Solver(g, 20)
        found = 0
        for _ in range(3000):
            d, s = bot_position(g, 0, rng, stop_when_loony=True)
            if (s == 4).all():
                continue
            sv = simple_value(d, s, g.line_box, g.box_lines, g.B, fast.W, fast.cnt, fast.memo, LOONY_R)
            if sv <= -1000 or g.L - d.sum() > 30:
                continue
            ok1, v1, _ = ref.solve(d, s, 50_000_000, F_PRUNE)
            if ok1:
                assert v1 == sv, (N, v1, sv)
                found += 1
            if found >= (15 if quick else 60):
                break
        n_simple += found
    ok(f"chain theory + de-dup == plain search ({n_cmp} bot endgames, {n_simple} simple chain/loop positions)")

    g = geometry(5)
    book = LossBook(None)
    pool = SearchPool(g, 1, max_nodes=g.L + 4)
    for m in rng.sample(range(g.L), 14):
        if not pool.over(0) and pool.R_drawn[0, m] == 0:
            pool.apply(0, m)
    d = pool.R_drawn[0].copy()
    move = int(rng.choice(list(np.flatnonzero(d == 0))))
    book.store(g, d, move)
    for t in range(8):
        dt = np.zeros_like(d)
        dt[g.line_perm[t]] = d
        assert book.lookup(g, dt) == int(g.line_perm[t][move])
    ok("loss book finds the same position under all 8 symmetries")

    pool = SearchPool(g, 1, max_nodes=800 * g.L, K=8, leaf_solve=0)
    pool.start_search(0, 800, noise=True)
    while len(pool.done_games()) == 0:
        n = pool.select()
        if n:
            pool.expand(n, np.random.randn(n, g.L).astype(np.float32), np.zeros(n, np.float32))
    vis, tot = pool.policy(0)
    assert tot == pool.TI[0, 0, I_N] - 1 and not (pool.TB[0, :pool.t_used[0], B_STATE] == ST_PEND).any()
    ok("batched MCTS with virtual loss: visit accounting consistent")

    if torch is not None:
        model = DABNet(16, 2).eval()
        ev = NetEvaluator(model, g, torch.device("cpu"))
        pool = SearchPool(g, 1, max_nodes=200 * g.L, K=4, leaf_solve=10)
        m, info = ai_think(pool, ev, g, 0.3, 10000, None)
        assert pool.R_drawn[0, m] == 0
        ok(f"network + ai_think on 5x5 ({info['src']}, {info.get('visits', 0)} visits)")
    print("ALL TESTS PASSED")


def bot_position(geo, rem, rng, stop_when_loony=False):
    pool = SearchPool(geo, 1, max_nodes=geo.L + 4)
    stack = np.zeros(geo.L + 4, np.int64)
    cand = np.zeros(geo.L, np.int64)
    nb_seed(rng.randrange(1 << 30))
    while pool.remaining(0) > rem:
        if stop_when_loony and not (pool.R_sides[0] == 3).any() and \
                count_safe_lines(pool.R_drawn[0], pool.R_sides[0], geo.line_box, geo.L) == 0:
            if rng.random() < 0.5:
                break
        pool.apply(0, int(heuristic_move(pool.R_drawn[0], pool.R_sides[0], geo.line_box, geo.box_lines,
                                         geo.L, geo.B, stack, cand)))
    return pool.R_drawn[0].copy(), pool.R_sides[0].copy()


def bench(args):
    geo = geometry(args.board)
    board_defaults(args, geo)
    rng = random.Random(1)
    print(f"== exact solver on realistic {geo.N}x{geo.N} positions (heuristic-bot games), budget {args.root_budget:,} nodes")
    sol = Solver(geo, args.tt_bits)
    sol.solve(*bot_position(geo, 12, rng), budget=10 ** 6)  # compile
    for rem in args.bench_lines:
        for flags, name in ((F_PRUNE, "plain"), (F_ALL, "chain-theory")):
            times, solved, nodes = [], 0, 0
            r2 = random.Random(rem)
            for _ in range(args.trials):
                d, s = bot_position(geo, rem, r2)
                t = time.time()
                good, _, _ = sol.solve(d, s, args.root_budget, flags)
                times.append(time.time() - t)
                solved += good
                nodes += sol.nodes_used
            print(f"  {rem:2d} lines left [{name:12s}]: solved {solved}/{args.trials}  avg {np.mean(times):.3f}s  "
                  f"max {np.max(times):.3f}s  ({nodes / max(1e-9, sum(times)):,.0f} nodes/s)", flush=True)
    if torch is None:
        return
    device = pick_device(args.device)
    model = DABNet(args.channels, args.blocks).to(device).eval()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        model.to(memory_format=torch.channels_last)
    ev = NetEvaluator(model, geo, device)
    print(f"== network {args.channels}x{args.blocks} on {device}")
    for bs in (64, 256, 512, 1024):
        X = np.random.rand(bs, C_IN, geo.S, geo.S).astype(np.float32)
        ev(X)
        ev(X)
        t = time.time()
        reps = 5
        for _ in range(reps):
            ev(X)
        dt = (time.time() - t) / reps
        print(f"  batch {bs:4d}: {dt * 1000:7.1f} ms  ->  {bs / dt:,.0f} positions/s", flush=True)
    if args.seconds <= 0:
        return
    nthr = args.solver_threads or os.cpu_count()
    print(f"== self-play throughput: {args.games} parallel games, {args.seconds:.0f}s, {nthr} solver threads")
    rep = Replay(geo, 100_000)
    sp = SelfPlay(geo, args, ev, rep, nthr)
    t = time.time()
    rounds = 0
    while time.time() - t < args.seconds:
        sp.step()
        rounds += 1
    dt = time.time() - t
    sp.close()
    print(f"  {sp.st['sims'] / dt:,.0f} simulations/s | {rounds / dt:.1f} batches/s | games finished "
          f"{sp.st['games']} | exact-solved moves {sp.st['solved']}")


# =====================================================================
# 17. COMMAND LINE
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="Nightmare Dots & Boxes (RL + MCTS + exact solver)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--board", type=int, default=6, help="boxes per side (5 or 6)")
        p.add_argument("--device", default="auto")
        p.add_argument("--tt-bits", type=int, default=22, help="solver hash table = 2^bits entries")
        p.add_argument("--seed", type=int, default=42)

    tr = sub.add_parser("train")
    common(tr)
    tr.add_argument("--out", default=None)
    tr.add_argument("--hours", type=float, default=8.3, help="stop and save before this many hours")
    tr.add_argument("--games", type=int, default=256, help="self-play games searched in parallel")
    tr.add_argument("--sims-full", type=int, default=None)
    tr.add_argument("--sims-fast", type=int, default=None)
    tr.add_argument("--p-full", type=float, default=0.25, help="fraction of moves with full search")
    tr.add_argument("--temp-moves", type=int, default=None)
    tr.add_argument("--temperature", type=float, default=1.0)
    tr.add_argument("--root-solve-lines", type=int, default=None)
    tr.add_argument("--root-budget", type=int, default=300_000)
    tr.add_argument("--loony-budget", type=int, default=2_000_000, help="solver nodes once no safe move is left")
    tr.add_argument("--leaf-solve-lines", type=int, default=None)
    tr.add_argument("--leaf-budget", type=int, default=5_000)
    tr.add_argument("--dir-alpha", type=float, default=None)
    tr.add_argument("--c-puct", type=float, default=1.6)
    tr.add_argument("--channels", type=int, default=64)
    tr.add_argument("--blocks", type=int, default=8)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--lr-floor", type=float, default=0.15, help="cosine decay ends at lr * this")
    tr.add_argument("--wd", type=float, default=1e-4)
    tr.add_argument("--batch", type=int, default=512)
    tr.add_argument("--train-ratio", type=float, default=3.0, help="training samples per new position")
    tr.add_argument("--replay", type=int, default=None)
    tr.add_argument("--min-replay", type=int, default=20_000)
    tr.add_argument("--positions-per-iter", type=int, default=8000)
    tr.add_argument("--save-minutes", type=float, default=15.0)
    tr.add_argument("--first-eval-minutes", type=float, default=12.0)
    tr.add_argument("--eval-minutes", type=float, default=35.0)
    tr.add_argument("--reserve-minutes", type=float, default=4.0, help="kept free at the end for final arena + save")
    tr.add_argument("--solver-threads", type=int, default=0, help="0 = cpu_count / procs-sharing-cpu")
    tr.add_argument("--procs-sharing-cpu", type=int, default=1, help="training processes sharing this machine")
    tr.add_argument("--eval-games", type=int, default=64)
    tr.add_argument("--eval-sims", type=int, default=100)
    tr.add_argument("--resume-from", default=None, help="folder with latest.pt/replay.npz from a previous session")
    tr.add_argument("--init-from", default=None, help="start weights from another model (e.g. 5x5 best.pt)")
    tr.add_argument("--hard-file", default=None, help="hard_N.npz produced by play/learn")
    tr.add_argument("--w-value", type=float, default=1.0)
    tr.add_argument("--w-own", type=float, default=0.25)
    tr.add_argument("--max-iters", type=int, default=0)

    def play_args(p):
        p.add_argument("--model", required=True)
        p.add_argument("--think", type=float, default=8.0, help="seconds per AI move")
        p.add_argument("--first", choices=("human", "ai", "random"), default="human")
        p.add_argument("--leaves", type=int, default=16, help="leaves per network batch")
        p.add_argument("--max-nodes", type=int, default=3_000_000)
        p.add_argument("--solve-lines", type=int, default=40, help="try exact solve at or below this many lines")
        p.add_argument("--solve-nodes", type=int, default=3_000_000)
        p.add_argument("--leaf-solve-lines", type=int, default=14)
        p.add_argument("--leaf-budget", type=int, default=20_000)
        p.add_argument("--learn-think", type=float, default=None)
        p.add_argument("--data-dir", default="dab_data")
        p.add_argument("--no-learn", action="store_true", help="do not analyse games the AI failed to win")
        p.add_argument("--learn-margin", type=int, default=1,
                       help="also analyse games the AI won by this margin or less")

    pl = sub.add_parser("play")
    common(pl)
    play_args(pl)
    le = sub.add_parser("learn")
    common(le)
    play_args(le)
    be = sub.add_parser("bench")
    common(be)
    be.add_argument("--channels", type=int, default=64)
    be.add_argument("--blocks", type=int, default=8)
    be.add_argument("--root-budget", type=int, default=300_000)
    be.add_argument("--games", type=int, default=256)
    be.add_argument("--seconds", type=float, default=30.0)
    be.add_argument("--trials", type=int, default=5)
    be.add_argument("--solver-threads", type=int, default=0)
    be.add_argument("--bench-lines", type=int, nargs="+", default=[20, 28, 36, 44])
    for name, val in (("p_full", 0.25), ("temperature", 1.0), ("leaf_budget", 5000), ("c_puct", 1.6),
                      ("sims_full", None), ("sims_fast", None), ("root_solve_lines", None), ("loony_budget", 2_000_000),
                      ("leaf_solve_lines", None), ("temp_moves", None), ("dir_alpha", None), ("replay", None)):
        be.set_defaults(**{name: val})
    te = sub.add_parser("test")
    te.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.cmd == "test":
        run_tests(args.quick)
        return
    seed_everything(args.seed)
    if args.cmd == "train":
        args.out = args.out or f"runs/b{args.board}"
        train(args)
    elif args.cmd == "bench":
        bench(args)
    else:
        if args.learn_think is None:
            args.learn_think = 4 * args.think
        play(args) if args.cmd == "play" else learn_cmd(args)


if __name__ == "__main__":
    main()
