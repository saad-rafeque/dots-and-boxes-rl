#!/usr/bin/env python3
# =====================================================================
#  NIGHTMARE DOTS & BOXES - GUI  (human vs AI, 5x5 or 6x6 boxes)
#
#  Put this file next to nightmare_dab.py and the trained models:
#      models/best_5x5.pt   models/best_6x6.pt      (from the Kaggle output)
#  then run:   python dab_gui.py
#
#  Needs: pip install numpy numba torch        (tkinter ships with Python)
#  Click a gap between two dots to draw a line. Complete a box -> move again.
# =====================================================================

import os
import sys
import json
import time
import queue
import random
import threading
import contextlib
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nightmare_dab as D  # noqa: E402

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:          # engine part still importable for tests
    tk = ttk = messagebox = None

LEARN_MARGIN = 1             # also study games the AI won by this margin or less
MODES = {                    # label -> mode
    "You vs AI": "play",
    "DQN vs AlphaZero (watch)": "watch",
}
WATCH_PAUSE_MS = 700         # so an instant DQN move is still visible
ENGINE_NAME = {"dqn": "DQN", "az": "AlphaZero"}
OPPONENTS = {                # label -> engine kind
    "AlphaZero (strongest)": "az",
    "DQN (5x5 only)": "dqn",
}
LEVELS = {                   # label -> seconds of thinking per AI move
    "Nightmare (20 s)": 20.0,
    "Very hard (10 s)": 10.0,
    "Hard (5 s)": 5.0,
    "Quick (2 s)": 2.0,
}
COL = dict(bg="#f7f7f5", panel="#ffffff", ink="#1f2937", muted="#6b7280", dot="#111827",
           grid="#e5e7eb", hover="#9ca3af", you="#2563eb", you_fill="#dbeafe",
           ai="#dc2626", ai_fill="#fee2e2", ai_last="#f97316")


# =====================================================================
#  ENGINE  (no tkinter here)
# =====================================================================

def find_model(N):
    names = [f"models/best_{N}x{N}.pt", f"best_{N}x{N}.pt", f"runs/b{N}/best.pt",
             f"models/final_{N}x{N}.pt", f"runs/b{N}/final.pt"]
    for n in names:
        p = os.path.join(HERE, n)
        if os.path.exists(p):
            return p
    return None


def find_dqn_model(N):
    """The DQN was trained on 5x5 only."""
    if N != 5:
        return None
    for n in ("models/dqn_a.pt", "models/dqn_b.pt", "DQN/models/dqn_a.pt"):
        p = os.path.join(HERE, n)
        if os.path.exists(p):
            return p
    return None


def load_dqn(path):
    """Returns (net, encoder, line_flat_tensor, q_from, geometry) or raises."""
    sys.path.insert(0, os.path.join(HERE, "DQN", "dqn_code"))
    import torch as T
    import game as DQG
    from model import DuelingQNet, q_from
    ck = T.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {"channels": 64, "blocks": 6})
    net = DuelingQNet(cfg["channels"], cfg["blocks"])
    net.load_state_dict(ck["net"])
    net.eval()
    g = DQG.Geometry(int(ck.get("board", 5)))
    lf = T.as_tensor((g.line_cell[:, 0] * g.S + g.line_cell[:, 1]).astype(np.int64))
    return net, DQG.Encoder(g, 1), lf, q_from, g, ck


class UniformEval:
    """Fallback when no trained model / no torch: search + exact solver only (much weaker)."""

    def __init__(self, geo):
        self.L = geo.L

    def __call__(self, X):
        n = X.shape[0]
        return np.zeros((n, self.L), np.float32), np.zeros(n, np.float32)


class Engine:
    def __init__(self, N, data_dir, kind="az"):
        self.N = N
        self.kind = kind
        self.geo = D.geometry(N)
        self.args = SimpleNamespace(max_nodes=3_000_000, leaves=16, tt_bits=22, leaf_solve_lines=14,
                                    leaf_budget=20_000, solve_lines=40, solve_nodes=3_000_000,
                                    learn_think=8.0)
        self.pool = D.make_play_pool(self.geo, self.args)
        self.model_path = find_model(N)
        self.trained = False
        self.ev = UniformEval(self.geo)
        self.note = "no trained model found - AI is much weaker"
        if self.model_path and D.torch is not None:
            dev = D.pick_device("auto")
            model, ck = D.load_net(self.model_path, dev)
            D.torch.set_num_threads(max(1, os.cpu_count() or 1))
            self.ev = D.NetEvaluator(model, self.geo, dev)
            self.trained = True
            st = ck.get("state", {})
            self.note = (f"model {os.path.basename(self.model_path)} "
                         f"({st.get('games', 0):,} self-play games, {st.get('hours', 0):.1f} h)")
        elif self.model_path:
            self.note = "torch not installed - AI is much weaker (pip install torch)"
        self.dqn = None
        if kind == "dqn":
            p = find_dqn_model(N)
            if p is None:
                self.kind = "az"
                self.note += "  |  no DQN model for this board - using AlphaZero"
            else:
                try:
                    (self.dqn, self.dqn_enc, self.dqn_lf,
                     self.dqn_qfrom, self.dqn_geo, ck) = load_dqn(p)
                    st = ck.get("state", {})
                    self.note = (f"DQN  {os.path.basename(p)}  "
                                 f"({st.get('games', 0):,} self-play games, "
                                 f"{st.get('updates', 0):,} updates)  -  plays instantly, "
                                 f"no search, no solver")
                except Exception as ex:
                    self.kind = "az"
                    self.note += f"  |  DQN failed to load ({type(ex).__name__}) - using AlphaZero"
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        self.book = D.LossBook(os.path.join(data_dir, f"book_{N}.pkl"))
        self.hard_path = os.path.join(data_dir, f"hard_{N}.npz")

    def warmup(self):
        g = self.geo
        self.pool.solver.solve(np.ones(g.L, np.uint8), np.full(g.B, 4, np.int8), 1000)
        self.pool.new_game(0)
        D.ai_think(self.pool, self.ev, self.geo, 0.2, 10_000, None, solve_lines=10, min_sims=8)
        self.pool.new_game(0)

    def think(self, seconds):
        if self.kind == "dqn":
            return self.think_dqn()
        return D.ai_think(self.pool, self.ev, self.geo, seconds, self.args.solve_nodes, self.book,
                          solve_lines=self.args.solve_lines)

    def think_dqn(self):
        """One forward pass, pick the highest-scoring legal line. No search, no solver -
        this is what a value-based agent actually is."""
        import torch as T
        p = self.pool
        s1, s2 = p.scores(0)
        drawn = np.ascontiguousarray(p.R_drawn[0])[None]
        sides = np.ascontiguousarray(p.R_sides[0])[None]
        owner = np.ascontiguousarray(p.R_owner[0])[None]
        t0 = time.time()
        X = self.dqn_enc(drawn, sides, owner, np.array([[s1, s2]], np.int32),
                         np.array([p.mover(0)], np.int8))
        with T.inference_mode():
            a, v = self.dqn(T.from_numpy(X), self.dqn_lf)
            q = self.dqn_qfrom(a.float(), v.float(), T.from_numpy(drawn == 0))
        m = int(q.argmax(1).item())
        pi = np.zeros(self.geo.L, np.float32)
        pi[m] = 1.0
        # same key set every other source returns, so the GUI never sees a hole
        return m, {"src": "dqn", "q": float(q[0, m].item()), "visits": 0,
                   "secs": time.time() - t0, "pi": pi}

    def learn(self, rec, seconds, progress=None):
        """Deep re-analysis of one game. Every position where a deeper search disagrees with
        what was played is stored in the loss book, so that exact position (and its 8 mirrored
        versions) never gets the same reply again."""
        self.args.learn_think = seconds
        lines = []
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            changed = D.learn_from_game(rec, self.geo, self.ev, self.args, self.book, self.hard_path,
                                        progress=progress, log=lines.append)
        return changed, len(self.book.data), lines


# =====================================================================
#  GUI
# =====================================================================

class App:
    def __init__(self, root):
        self.root = root
        self.data_dir = os.path.join(HERE, "dab_data")
        self.engines = {}
        self.watch = {}                 # mover -> Engine, empty unless watching
        self.think_inflight = False     # exactly one engine may be thinking at a time
        self.jobs = queue.Queue()
        self.busy = None             # "loading" | "thinking" | "learning" | "replay" | None
        self.token = 0
        self.rec = None
        self.last_rec = None
        self.highlight = []
        self.hover = None
        self.replay_state = None
        self.game_over = True
        self.think_started = 0.0
        self.highlight_owner = "ai"
        self.learning = False
        self.learn_note = ""

        root.title("Nightmare Dots & Boxes")
        root.configure(bg=COL["bg"])
        root.minsize(620, 760)
        style = ttk.Style()
        style.configure("TFrame", background=COL["bg"])
        style.configure("TLabel", background=COL["bg"], foreground=COL["ink"])
        style.configure("TRadiobutton", background=COL["bg"])

        bar = ttk.Frame(root, padding=(12, 10, 12, 4))
        bar.pack(fill="x")
        self.board_var = tk.IntVar(value=5)
        self.first_var = tk.StringVar(value="You")
        self.level_var = tk.StringVar(value=list(LEVELS)[0])
        self.opp_var = tk.StringVar(value=list(OPPONENTS)[0])
        self.mode_var = tk.StringVar(value=list(MODES)[0])
        ttk.Label(bar, text="Mode").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.mode_cb = ttk.Combobox(bar, textvariable=self.mode_var, values=list(MODES),
                                    width=24, state="readonly")
        self.mode_cb.grid(row=1, column=1, columnspan=3, sticky="w", padx=(6, 0), pady=(6, 0))
        ttk.Label(bar, text="Board").grid(row=0, column=0, sticky="w")
        self.rb5 = ttk.Radiobutton(bar, text="5 x 5", variable=self.board_var, value=5)
        self.rb6 = ttk.Radiobutton(bar, text="6 x 6", variable=self.board_var, value=6)
        self.rb5.grid(row=0, column=1, padx=(6, 0))
        self.rb6.grid(row=0, column=2, padx=(4, 14))
        ttk.Label(bar, text="First move").grid(row=0, column=3, sticky="w")
        self.first_cb = ttk.Combobox(bar, textvariable=self.first_var, values=["You", "AI", "Random"],
                                     width=8, state="readonly")
        self.first_cb.grid(row=0, column=4, padx=(6, 14))
        ttk.Label(bar, text="Opponent").grid(row=0, column=5, sticky="w")
        self.opp_cb = ttk.Combobox(bar, textvariable=self.opp_var, values=list(OPPONENTS),
                                   width=20, state="readonly")
        self.opp_cb.grid(row=0, column=6, padx=(6, 14))
        ttk.Label(bar, text="AI").grid(row=0, column=7, sticky="w")
        self.level_cb = ttk.Combobox(bar, textvariable=self.level_var, values=list(LEVELS), width=16,
                                     state="readonly")
        self.level_cb.grid(row=0, column=8, padx=(6, 0))

        btns = ttk.Frame(root, padding=(12, 4, 12, 4))
        btns.pack(fill="x")
        self.new_btn = ttk.Button(btns, text="New Game", command=self.new_game)
        self.new_btn.pack(side="left")
        self.replay_btn = ttk.Button(btns, text="Replay Last Game", command=self.toggle_replay,
                                     state="disabled")
        self.replay_btn.pack(side="left", padx=8)
        self.stats_lbl = ttk.Label(btns, text="", foreground=COL["muted"])
        self.stats_lbl.pack(side="right")

        score = tk.Frame(root, bg=COL["panel"], highlightthickness=1, highlightbackground=COL["grid"])
        score.pack(fill="x", padx=12, pady=6)
        self.you_lbl = tk.Label(score, text="YOU  0", font=("Segoe UI", 22, "bold"), fg=COL["you"],
                                bg=COL["panel"])
        self.you_lbl.pack(side="left", padx=16, pady=8)
        self.ai_lbl = tk.Label(score, text="0  AI", font=("Segoe UI", 22, "bold"), fg=COL["ai"],
                               bg=COL["panel"])
        self.ai_lbl.pack(side="right", padx=16, pady=8)
        self.turn_lbl = tk.Label(score, text="", font=("Segoe UI", 12), fg=COL["ink"], bg=COL["panel"])
        self.turn_lbl.pack(expand=True)

        self.canvas = tk.Canvas(root, width=600, height=600, bg=COL["panel"], highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=12, pady=(0, 6))
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_move)
        self.canvas.bind("<Leave>", lambda e: self.set_hover(None))
        self.canvas.bind("<Configure>", lambda e: self.redraw())

        self.status = tk.Label(root, text="", anchor="w", fg=COL["muted"], bg=COL["bg"], font=("Segoe UI", 10))
        self.status.pack(fill="x", padx=12, pady=(0, 10))

        self.board_var.trace_add("write", lambda *a: self.new_game() if (self.busy is None or self.watch) else None)
        self.opp_var.trace_add("write", lambda *a: self.new_game() if (self.busy is None or self.watch) else None)
        self.mode_var.trace_add("write", lambda *a: self.new_game() if (self.busy is None or self.watch) else None)
        self.update_stats()
        self.root.after(50, self.poll)
        self.new_game()

    # ------------------------------------------------------------ engine / state
    @property
    def N(self):
        return self.board_var.get()

    @property
    def opp_kind(self):
        """DQN only exists for 5x5 - fall back rather than serve a bad opponent."""
        k = OPPONENTS.get(self.opp_var.get(), "az")
        if k == "dqn" and find_dqn_model(self.N) is None:
            return "az"
        return k

    @property
    def watching(self):
        """Watch mode needs both engines, and the DQN only exists for 5x5."""
        return (MODES.get(self.mode_var.get()) == "watch"
                and find_dqn_model(self.N) is not None)

    def engine(self):
        if self.watch:
            return self.watch[1]
        return self.engines.get((self.N, self.opp_kind))

    def mover_engine(self):
        """Whose brain decides the move on the board right now."""
        if self.watch:
            return self.watch[self.engine().pool.mover(0)]
        return self.engine()

    def all_pools(self):
        """Every pool that must be kept on the same board."""
        if self.watch:
            return [e.pool for e in dict.fromkeys(self.watch.values())]
        e = self.engine()
        return [e.pool] if e else []

    def state(self):
        """(geo, drawn, owner, s1, s2, mover) of what is on screen."""
        if self.replay_state is not None:
            return self.replay_state
        e = self.engine()
        if e is None:
            geo = D.geometry(self.N)
            return geo, np.zeros(geo.L, np.uint8), np.zeros(geo.B, np.int8), 0, 0, 1
        p = e.pool
        s1, s2 = p.scores(0)
        return e.geo, p.R_drawn[0], p.R_owner[0], s1, s2, p.mover(0)

    def set_busy(self, what):
        self.busy = what
        # while two engines play each other nothing is waiting on the user, so the
        # controls stay live and New Game can interrupt the match at any point
        idle = what is None or (bool(self.watch) and what == "thinking")
        for w in (self.rb5, self.rb6):
            w.configure(state="normal" if idle else "disabled")
        for w in (self.first_cb, self.opp_cb, self.mode_cb):
            w.configure(state="readonly" if idle else "disabled")
        self.opp_cb.configure(state="disabled" if self.watching else self.opp_cb["state"])
        # the thinking-time setting means nothing to the DQN: it never searches.
        # In watch mode it still controls how long AlphaZero thinks.
        dqn_only = self.opp_kind == "dqn" and not self.watching
        self.level_cb.configure(state="disabled" if (dqn_only or not idle) else "readonly")
        self.new_btn.configure(state="normal" if idle else "disabled")
        if what == "replay":
            self.replay_btn.configure(state="normal", text="Stop Replay")
        else:
            self.replay_btn.configure(text="Replay Last Game",
                                      state="normal" if (idle and self.last_rec) else "disabled")

    def run_bg(self, kind, fn, *a):
        tok = self.token

        def work():
            try:
                res = fn(*a)
                self.jobs.put((kind, tok, res, None))
            except Exception as ex:          # show errors instead of dying silently
                self.jobs.put((kind, tok, None, ex))
        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------ game flow
    def new_game(self):
        if self.busy is not None and not self.watch:
            return
        self.token += 1
        self.think_inflight = False      # results from the old game are discarded by token
        self.highlight = []
        self.game_over = True
        self.watch = {}
        if MODES.get(self.mode_var.get()) == "watch":
            if not self.watching:
                self.status.configure(
                    text="The DQN was only trained on 5x5 - switch the board to 5 x 5 to watch "
                         "the two engines play.")
            else:
                for k in ("dqn", "az"):
                    if (self.N, k) not in self.engines:
                        self.set_busy("loading")
                        self.status.configure(
                            text=f"Loading the {ENGINE_NAME[k]} for the match "
                                 f"(first start compiles the engine, ~1 min)...")
                        self.redraw()

                        def load(N=self.N, k=k):
                            e = Engine(N, self.data_dir, k)
                            e.warmup()
                            return e
                        self.run_bg("loaded", load)
                        return
                dqn_e = self.engines[(self.N, "dqn")]
                az_e = self.engines[(self.N, "az")]
                pick = self.first_var.get()
                dqn_first = pick == "You" or (pick == "Random" and random.random() < 0.5)
                self.watch = {1: dqn_e, 2: az_e} if dqn_first else {1: az_e, 2: dqn_e}
                for p in self.all_pools():
                    p.new_game(0)
                self.rec = {"board": self.N, "human": 0, "moves": [], "ai": [], "watch": True,
                            "p1": self.watch[1].kind, "p2": self.watch[2].kind}
                self.game_over = False
                self.status.configure(
                    text=f"{ENGINE_NAME[self.watch[1].kind]} moves first (blue), "
                         f"{ENGINE_NAME[self.watch[2].kind]} second (red). Press New Game to stop.")
                self.redraw()
                self.after_move()
                return
        N, kind = self.N, self.opp_kind
        if kind == "az" and OPPONENTS.get(self.opp_var.get()) == "dqn":
            self.status.configure(text=f"The DQN was only trained on 5x5 - "
                                       f"playing {N}x{N} against AlphaZero instead.")
        if (N, kind) not in self.engines:
            self.set_busy("loading")
            self.status.configure(
                text=f"Loading the {N}x{N} "
                     f"{'DQN' if kind == 'dqn' else 'AlphaZero'} "
                     f"(first start compiles the engine, ~1 min)...")
            self.redraw()

            def load(N=N, kind=kind):
                e = Engine(N, self.data_dir, kind)
                e.warmup()
                return e
            self.run_bg("loaded", load)
            return
        e = self.engines[(N, kind)]
        e.pool.new_game(0)
        first = self.first_var.get()
        human = {"You": 1, "AI": 2}.get(first, random.choice([1, 2]))
        self.rec = {"board": N, "human": human, "moves": [], "ai": []}
        self.game_over = False
        self.status.configure(text=e.note)
        self.redraw()
        self.after_move()

    def after_move(self):
        e = self.engine()
        p = e.pool
        self.update_scores()
        if p.over(0):
            self.finish_game()
            return
        if self.think_inflight:
            return                       # a think is already running for this board
        if self.watch:
            me = self.mover_engine()
            self.set_busy("thinking")
            self.turn_lbl.configure(
                text=f"{ENGINE_NAME[me.kind]} is thinking...",
                fg=COL["you"] if p.mover(0) == 1 else COL["ai"])
            self.think_started = time.time()
            self.think_inflight = True
            self.run_bg("move", me.think, LEVELS[self.level_var.get()])
            return
        if p.mover(0) == self.rec["human"]:
            self.set_busy(None)
            self.turn_lbl.configure(text="Your turn", fg=COL["you"])
        else:
            self.set_busy("thinking")
            self.turn_lbl.configure(text="AI is thinking...", fg=COL["ai"])
            self.think_started = time.time()
            self.think_inflight = True
            self.run_bg("move", e.think, LEVELS[self.level_var.get()])

    def human_move(self, m):
        if self.watch:
            return                       # nobody is playing by hand in a watch match
        e = self.engine()
        p = e.pool
        if self.game_over or self.busy is not None or p.mover(0) != self.rec["human"] or p.R_drawn[0, m]:
            return
        k = p.apply(0, m)
        self.rec["moves"].append(int(m))
        if self.highlight and self.highlight_owner == "ai":
            self.highlight = []
        self.highlight_owner = "human"
        self.hover = None
        self.status.configure(text="Box! Move again." if k else "")
        self.redraw()
        self.after_move()

    def ai_move(self, m, info):
        e = self.engine()
        p = e.pool
        if getattr(self, "highlight_owner", None) != "ai":
            self.highlight = []
        mover = p.mover(0)
        who = self.mover_engine() if self.watch else e
        self.highlight_owner = "human" if (self.watch and mover == 1) else "ai"
        k = 0
        for pl in self.all_pools():      # both engines stay on the same board
            k = pl.apply(0, m)
        self.rec["moves"].append(int(m))
        self.rec["ai"].append(info["src"])
        self.highlight.append(m)
        src = info.get("src", "?")
        conf = 50 * (1 + max(-1.0, min(1.0, float(info.get("q", 0.0)))))
        tag = f"{ENGINE_NAME[who.kind]} - " if self.watch else ""
        if src == "exact":
            txt = f"AI: exact endgame solve, it nets {info.get('value', 0):+d} of the remaining boxes"
        elif src == "book":
            txt = "AI: move from its memory of lost games"
        elif src == "dqn":
            boxes = float(info.get("q", 0.0)) * e.geo.B
            txt = (f"DQN: instant move, no search. It expects {boxes:+.1f} boxes "
                   f"from here, win confidence {conf:.0f}%")
        else:
            txt = (f"AI: {info.get('visits', 0):,} positions searched in "
                   f"{info.get('secs', 0.0):.1f}s, win confidence {conf:.0f}%")
        again = "  (box - moves again)" if k else ""
        self.status.configure(text=tag + txt + again)
        self.redraw()
        if self.watch:
            self.root.after(WATCH_PAUSE_MS, self.after_move)   # slow enough to follow
        elif not p.over(0) and p.mover(0) != self.rec["human"]:
            self.root.after(350, self.after_move)       # let the human see each AI line
        else:
            self.after_move()

    def finish_game(self):
        e = self.engine()
        s1, s2 = e.pool.scores(0)
        human = self.rec["human"]
        you, ai = (s1, s2) if human == 1 else (s2, s1)
        self.game_over = True
        if self.watch:
            you, ai = s1, s2             # player 1 and player 2, not human and AI
        self.rec.update(you=you, ai_score=ai, time=time.strftime("%Y-%m-%d %H:%M:%S"),
                        level=self.level_var.get(),
                        opponent="watch" if self.watch else e.kind)
        self.last_rec = dict(self.rec)
        try:
            with open(os.path.join(self.data_dir, f"games_{self.N}.jsonl"), "a") as f:
                f.write(json.dumps(self.rec) + "\n")
        except OSError:
            pass
        if self.watch:
            n1, n2 = ENGINE_NAME[self.watch[1].kind], ENGINE_NAME[self.watch[2].kind]
            if you > ai:
                res, col = f"{n1.upper()} WINS {you}-{ai}", COL["you"]
            elif you == ai:
                res, col = f"DRAW {you}-{ai}", COL["ink"]
            else:
                res, col = f"{n2.upper()} WINS {ai}-{you}", COL["ai"]
        elif you > ai:
            res, col = f"YOU WIN {you}-{ai}", COL["you"]
        elif you == ai:
            res, col = f"DRAW {you}-{ai}", COL["ink"]
        else:
            res, col = f"AI WINS {ai}-{you}", COL["ai"]
        self.turn_lbl.configure(text=res, fg=col)
        self.set_busy(None)
        self.update_stats()
        self.redraw()
        if not self.watch and e.kind == "az" and you >= ai - LEARN_MARGIN and not self.learning:
            # automatic: one game is enough - that line will not work a second time
            self.learning = True
            self.learn_note = "AI is studying this game..."
            self.status.configure(text=self.learn_note)
            self.run_bg("learned", e.learn, dict(self.rec),
                        max(4.0, LEVELS[self.level_var.get()]), self.on_learn_progress)

    # ------------------------------------------------------------ replay
    def toggle_replay(self):
        if self.busy == "replay":
            self.stop_replay()
            return
        if self.busy is not None or not self.last_rec:
            return
        geo = D.geometry(self.last_rec["board"])
        self.set_busy("replay")
        if self.last_rec["board"] != self.N:
            self.board_var.set(self.last_rec["board"])     # busy -> no new game is started
        self.replay_state = (geo, np.zeros(geo.L, np.uint8), np.zeros(geo.B, np.int8), 0, 0, 1)
        self.replay_sides = np.zeros(geo.B, np.int8)
        self.replay_i = 0
        self.highlight = []
        self.turn_lbl.configure(text="Replay", fg=COL["ink"])
        self.root.after(400, self.replay_step)

    def replay_step(self):
        if self.busy != "replay":
            return
        rec = self.last_rec
        geo, drawn, owner, s1, s2, mover = self.replay_state
        if self.replay_i >= len(rec["moves"]):
            self.status.configure(text=f"Replay finished: you {rec['you']} - {rec['ai_score']} AI")
            self.root.after(1500, self.stop_replay)
            return
        m = rec["moves"][self.replay_i]
        k = D.apply_move(drawn, self.replay_sides, owner, geo.line_box, m, mover)
        if mover == 1:
            s1 += k
        else:
            s2 += k
        who = "You" if mover == rec["human"] else "AI"
        self.highlight = [m]
        self.highlight_owner = "human" if who == "You" else "ai"
        if k == 0:
            mover = 3 - mover
        self.replay_state = (geo, drawn, owner, s1, s2, mover)
        self.replay_i += 1
        self.status.configure(text=f"Replay move {self.replay_i}/{len(rec['moves'])}: {who} "
                                   f"{geo.describe_line(m)}" + ("  +box" if k else ""))
        self.update_scores()
        self.redraw()
        self.root.after(450, self.replay_step)

    def stop_replay(self):
        if self.busy != "replay":
            return
        self.replay_state = None
        self.highlight = []
        self.set_busy(None)
        self.update_scores()
        self.turn_lbl.configure(text="Press New Game", fg=COL["ink"])
        self.redraw()

    # ------------------------------------------------------------ events
    def on_learn_progress(self, done, total, fixed):
        self.learn_note = (f"AI is studying this game: move {done + 1}/{total}, "
                           f"{fixed} fix{'es' if fixed != 1 else ''} found so far")

    def poll(self):
        try:
            while True:
                kind, tok, res, err = self.jobs.get_nowait()
                if err is not None:
                    self.learning = False
                    self.set_busy(None)
                    messagebox.showerror("Error", f"{type(err).__name__}: {err}")
                    continue
                if kind == "loaded":
                    self.engines[(res.N, res.kind)] = res
                    self.set_busy(None)
                    if res.N == self.N:
                        self.new_game()   # picks up watch mode again if that is selected
                elif kind == "move":
                    self.think_inflight = False
                    if tok == self.token:
                        m, info = res
                        self.ai_move(m, info)
                    elif not self.game_over:
                        # belonged to an interrupted game - restart the current one
                        self.root.after(10, self.after_move)
                elif kind == "learned":
                    self.learning = False
                    self.learn_note = ""
                    changed, book_n, _ = res
                    self.status.configure(
                        text=f"AI finished studying that game: {changed} move(s) corrected. "
                             f"It will not lose the same way again. Memory: {book_n} positions.")
                    self.update_stats()
        except queue.Empty:
            pass
        if self.busy == "thinking":
            nm = ENGINE_NAME[self.mover_engine().kind] if self.watch else "AI"
            self.turn_lbl.configure(text=f"{nm} is thinking... {time.time() - self.think_started:.0f}s")
        if self.learning and self.busy is None and self.learn_note:
            self.status.configure(text=self.learn_note)
        self.root.after(80, self.poll)

    def geometry_px(self, geo):
        w = max(self.canvas.winfo_width(), 200)
        h = max(self.canvas.winfo_height(), 200)
        size = min(w, h)
        margin = 36
        cell = (size - 2 * margin) / geo.N
        ox = (w - cell * geo.N) / 2
        oy = (h - cell * geo.N) / 2
        return ox, oy, cell

    def line_xy(self, geo, l, ox, oy, cell):
        r, c = geo.line_cell[l]           # grid coords: (2r, 2c+1) horizontal, (2r+1, 2c) vertical
        if r % 2 == 0:
            y = oy + (r // 2) * cell
            x0 = ox + (c // 2) * cell
            return x0, y, x0 + cell, y
        x = ox + (c // 2) * cell
        y0 = oy + (r // 2) * cell
        return x, y0, x, y0 + cell

    def nearest_line(self, x, y):
        geo, drawn, *_ = self.state()
        ox, oy, cell = self.geometry_px(geo)
        best, bd = None, cell * 0.28
        for l in range(geo.L):
            if drawn[l]:
                continue
            x0, y0, x1, y1 = self.line_xy(geo, l, ox, oy, cell)
            px = min(max(x, min(x0, x1)), max(x0, x1))
            py = min(max(y, min(y0, y1)), max(y0, y1))
            d = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            if d < bd:
                best, bd = l, d
        return best

    def can_play(self):
        e = self.engine()
        return (e is not None and not self.game_over and self.busy is None and self.rec is not None
                and e.pool.mover(0) == self.rec["human"])

    def on_move(self, ev):
        self.set_hover(self.nearest_line(ev.x, ev.y) if self.can_play() else None)

    def set_hover(self, l):
        if l != self.hover:
            self.hover = l
            self.redraw()

    def on_click(self, ev):
        if not self.can_play():
            return
        l = self.nearest_line(ev.x, ev.y)
        if l is not None:
            self.human_move(l)

    # ------------------------------------------------------------ drawing
    def update_scores(self):
        geo, drawn, owner, s1, s2, mover = self.state()
        rec = self.rec if self.replay_state is None else self.last_rec
        human = rec["human"] if rec else 1
        you, ai = (s1, s2) if human == 1 else (s2, s1)
        left = geo.B - s1 - s2
        if self.watch:
            self.you_lbl.configure(text=f"{ENGINE_NAME[self.watch[1].kind].upper()}  {s1}")
            self.ai_lbl.configure(text=f"{s2}  {ENGINE_NAME[self.watch[2].kind].upper()}")
        else:
            self.you_lbl.configure(text=f"YOU  {you}")
            self.ai_lbl.configure(text=f"{ai}  AI")
        if self.replay_state is None and rec and not self.game_over:
            self.root.title(f"Nightmare Dots & Boxes - {geo.N}x{geo.N} - you {you} : {ai} AI - {left} boxes left")

    def update_stats(self):
        w = d = l = 0
        for N in (5, 6):
            p = os.path.join(self.data_dir, f"games_{N}.jsonl")
            if os.path.exists(p):
                for line in open(p):
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if r.get("watch"):
                        continue
                    w += r["you"] > r["ai_score"]
                    d += r["you"] == r["ai_score"]
                    l += r["you"] < r["ai_score"]
        book_n = sum(len(e.book.data) for e in self.engines.values() if e.kind == "az")
        mem = f"  |  AI memory: {book_n} positions" if book_n else ""
        self.stats_lbl.configure(text=f"All games - you won {w}, drew {d}, AI won {l}{mem}")

    def redraw(self):
        cv = self.canvas
        cv.delete("all")
        geo, drawn, owner, s1, s2, mover = self.state()
        rec = self.rec if self.replay_state is None else self.last_rec
        human = rec["human"] if rec else 1
        ox, oy, cell = self.geometry_px(geo)
        N = geo.N
        for b in range(geo.B):
            if owner[b]:
                r, c = divmod(b, N)
                mine = owner[b] == human
                x0, y0 = ox + c * cell, oy + r * cell
                cv.create_rectangle(x0 + 3, y0 + 3, x0 + cell - 3, y0 + cell - 3, width=0,
                                    fill=COL["you_fill"] if mine else COL["ai_fill"])
                cv.create_text(x0 + cell / 2, y0 + cell / 2, text="YOU" if mine else "AI",
                               fill=COL["you"] if mine else COL["ai"],
                               font=("Segoe UI", max(9, int(cell / 6)), "bold"))
        hl = set(self.highlight)
        hl_ai = getattr(self, "highlight_owner", "ai") == "ai"
        for l in range(geo.L):
            x0, y0, x1, y1 = self.line_xy(geo, l, ox, oy, cell)
            if drawn[l]:
                if l in hl:
                    col = COL["ai_last"] if hl_ai else COL["you"]
                    cv.create_line(x0, y0, x1, y1, fill=col, width=6, capstyle="round")
                else:
                    cv.create_line(x0, y0, x1, y1, fill=COL["ink"], width=4, capstyle="round")
            elif l == self.hover:
                cv.create_line(x0, y0, x1, y1, fill=COL["you"], width=4, dash=(6, 4), capstyle="round")
            else:
                cv.create_line(x0, y0, x1, y1, fill=COL["grid"], width=2)
        rad = max(4, cell / 14)
        for r in range(N + 1):
            for c in range(N + 1):
                x, y = ox + c * cell, oy + r * cell
                cv.create_oval(x - rad, y - rad, x + rad, y + rad, fill=COL["dot"], width=0)
        if self.busy == "loading":
            cv.create_text(ox + cell * N / 2, oy + cell * N / 2, text="Loading AI...",
                           font=("Segoe UI", 18, "bold"), fill=COL["muted"])


def main():
    if tk is None:
        sys.exit("tkinter is missing - install the standard Python from python.org (it includes tkinter)")
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.2)
    except tk.TclError:
        pass
    app = App(root)
    def on_close():
        # a background think thread may still be inside torch; letting the
        # interpreter unwind around it prints an ugly "terminate called" abort,
        # so leave immediately once the window is gone
        app.token += 1
        try:
            root.destroy()
        finally:
            os._exit(0)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
