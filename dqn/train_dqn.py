#!/usr/bin/env python3
"""Dueling Double-DQN with n-step returns for Dots & Boxes.

  python train_dqn.py --board 5 --out runs/dqn5 --hours 8.5
  python train_dqn.py --eval-only --resume-from runs/dqn5

Design notes
------------
* Self-play: both sides act eps-greedily from the SAME online network, so the
  opponent improves with the agent.  Transitions are always stored from the
  point of view of the player to move.
* Two-player bootstrap: completing a box means you move again, so the target
  carries an explicit sign instead of assuming the turn alternated (buffer.py).
* Reward shaping: r = boxes captured by this move / B.  Signed and summed over a
  game this telescopes to the exact final margin / B, so the shaping is exact,
  not a heuristic -- it just moves the signal earlier, which is the single
  biggest problem a value-based method has on this game.
* Double DQN: the online net picks the bootstrap action, the target net scores
  it, which removes most of the max-operator optimism bias.
"""
import os, sys, time, json, math, signal, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import game as G
from game import Geometry, Batch, Encoder, sides_from_drawn, C_IN
from model import DuelingQNet, q_from
from buffer import Replay, NStepTracker


def log_to(path):
    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        try:
            with open(path, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass
    return log


def atomic_save(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


class Runner:
    def __init__(self, args):
        self.args = args
        self.geo = geo = Geometry(args.board)
        self.dev = torch.device(args.device if args.device != "auto" else
                                ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)

        self.net = DuelingQNet(args.channels, args.blocks).to(self.dev)
        self.tgt = DuelingQNet(args.channels, args.blocks).to(self.dev)
        self.tgt.load_state_dict(self.net.state_dict())
        self.tgt.eval()
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=args.lr, weight_decay=1e-4)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.dev.type == "cuda")
        self.line_flat = torch.as_tensor(
            (geo.line_cell[:, 0] * geo.S + geo.line_cell[:, 1]).astype(np.int64), device=self.dev)

        self.replay = Replay(geo, args.buffer)
        self.batch = Batch(geo, args.games, self.rng)
        self.tracker = NStepTracker(args.games, args.nstep, args.gamma)
        self.enc_play = Encoder(geo, args.games)
        self.enc_train = Encoder(geo, args.batch)
        self.state = dict(steps=0, updates=0, games=0, hours=0.0, best_vs_bot=-1.0)

    # ------------------------------------------------------------ act
    @torch.inference_mode()
    def q_of(self, drawn, sides, owner, score, mover, enc, net=None):
        X = enc(drawn, sides, owner, score, mover)
        x = torch.from_numpy(X).to(self.dev)
        legal = torch.from_numpy(drawn == 0).to(self.dev)
        with torch.autocast(device_type=self.dev.type, dtype=torch.float16,
                            enabled=self.dev.type == "cuda"):
            a, v = (net or self.net)(x, self.line_flat)
        return q_from(a.float(), v.float(), legal)

    def eps_now(self):
        a = self.args
        f = min(1.0, self.state["steps"] / max(1, a.eps_decay_steps))
        return a.eps_start + (a.eps_end - a.eps_start) * f

    def collect(self, n_steps):
        """Run n_steps of self-play across all parallel games."""
        b, geo, a = self.batch, self.geo, self.args
        self.net.eval()
        eps = self.eps_now()
        pending = {f: [] for f in Replay.FIELDS}
        for _ in range(n_steps):
            live = ~b.done
            if not live.any():
                b.reset(); [self.tracker.reset_game(g) for g in range(a.games)]
                continue
            q = self.q_of(b.drawn, b.sides, b.owner, b.score, b.mover, self.enc_play)
            greedy = q.argmax(1).cpu().numpy()
            legal = b.legal()
            explore = self.rng.random(a.games) < eps
            act = greedy.copy()
            for g in np.flatnonzero(explore & live):
                cand = np.flatnonzero(legal[g])
                act[g] = cand[self.rng.integers(len(cand))]
            snap = (b.drawn.copy(), b.owner.copy(), b.score.astype(np.int16).copy(), b.mover.copy())
            for g in np.flatnonzero(live):
                self.tracker.push(g, (snap[0][g], snap[1][g], snap[2][g], snap[3][g]),
                                  int(act[g]), 0.0, True)
            rew, same = b.step(act)
            for g in np.flatnonzero(live):
                self.tracker.buf[g][-1][2] = float(rew[g])
                self.tracker.buf[g][-1][3] = bool(same[g])
            nxt = (b.drawn, b.owner, b.score.astype(np.int16), b.mover)
            for g in np.flatnonzero(live):
                term = bool(b.done[g])
                # copy: these rows are about to be reset for finished games
                ns = (nxt[0][g].copy(), nxt[1][g].copy(), nxt[2][g].copy(), int(nxt[3][g]))
                for tr in self.tracker.pop_ready(g, ns, term):
                    for f in Replay.FIELDS:
                        pending[f].append(tr[f])
                if term:
                    self.state["games"] += 1
            self.state["steps"] += int(live.sum())
            fin = np.flatnonzero(b.done)
            if len(fin):
                b.reset(fin)
                for g in fin:
                    self.tracker.reset_game(int(g))
        if pending["action"]:
            self.replay.add_many(**{f: np.array(pending[f]) for f in Replay.FIELDS})

    # ------------------------------------------------------------ learn
    def update(self):
        a, geo = self.args, self.geo
        s = self.replay.sample(a.batch, self.rng)
        sides = sides_from_drawn(geo, s["drawn"])
        n_sides = sides_from_drawn(geo, s["n_drawn"])
        X = self.enc_train(s["drawn"], sides, s["owner"], s["score"], s["mover"])
        NX = self.enc_train(s["n_drawn"], n_sides, s["n_owner"], s["n_score"], s["n_mover"])
        x = torch.from_numpy(X.copy()).to(self.dev)
        nx = torch.from_numpy(NX.copy()).to(self.dev)
        legal = torch.from_numpy(s["drawn"] == 0).to(self.dev)
        n_legal = torch.from_numpy(s["n_drawn"] == 0).to(self.dev)
        act = torch.from_numpy(s["action"].astype(np.int64)).to(self.dev)
        r = torch.from_numpy(s["r_acc"]).to(self.dev)
        sign = torch.from_numpy(s["sign"].astype(np.float32)).to(self.dev)
        done = torch.from_numpy(s["done"].astype(np.float32)).to(self.dev)
        disc = torch.from_numpy((a.gamma ** s["k"]).astype(np.float32)).to(self.dev)

        with torch.no_grad():
            with torch.autocast(device_type=self.dev.type, dtype=torch.float16,
                                enabled=self.dev.type == "cuda"):
                na_on, nv_on = self.net(nx, self.line_flat)
                na_tg, nv_tg = self.tgt(nx, self.line_flat)
            best = q_from(na_on.float(), nv_on.float(), n_legal).argmax(1, keepdim=True)
            q_next = q_from(na_tg.float(), nv_tg.float(), n_legal).gather(1, best).squeeze(1)
            q_next = torch.where(n_legal.any(1), q_next, torch.zeros_like(q_next))
            y = r + (1.0 - done) * sign * disc * q_next
            y = y.clamp(-1.0, 1.0)

        self.net.train()
        with torch.autocast(device_type=self.dev.type, dtype=torch.float16,
                            enabled=self.dev.type == "cuda"):
            adv, v = self.net(x, self.line_flat)
        q = q_from(adv.float(), v.float(), legal).gather(1, act[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(q, y, beta=0.1)
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.state["updates"] += 1
        if self.state["updates"] % a.target_sync == 0:
            self.tgt.load_state_dict(self.net.state_dict())
        return float(loss.detach()), float(y.mean()), float(q.detach().mean())

    # ------------------------------------------------------------ eval
    @torch.inference_mode()
    def eval_vs_bot(self, n_games, seed=0):
        """Greedy DQN against the fixed heuristic bot. No search on either side."""
        geo, a = self.geo, self.args
        n_games += n_games % 2
        rng = np.random.default_rng(seed)
        b = Batch(geo, n_games, rng)
        enc = Encoder(geo, n_games)
        side = np.where(np.arange(n_games) % 2 == 0, 1, 2).astype(np.int8)
        self.net.eval()
        while not b.done.all():
            ours = (b.mover == side) & ~b.done
            theirs = (~ours) & ~b.done
            act = np.zeros(n_games, np.int64)
            if theirs.any():
                act[theirs] = b.heuristic()[theirs]
            if ours.any():
                q = self.q_of(b.drawn, b.sides, b.owner, b.score, b.mover, enc)
                act[ours] = q.argmax(1).cpu().numpy()[ours]
            act = np.where(b.done, 0, act)
            b.step(act)
        mg = np.where(side == 1, b.margin(), -b.margin())
        return dict(games=n_games, wins=int((mg > 0).sum()), draws=int((mg == 0).sum()),
                    losses=int((mg < 0).sum()), score=float(((mg > 0).sum() + 0.5 * (mg == 0).sum()) / n_games),
                    avg_margin=float(mg.mean()))

    # ------------------------------------------------------------ io
    def save(self, out, tag):
        atomic_save(dict(net=self.net.state_dict(), tgt=self.tgt.state_dict(),
                         opt=self.opt.state_dict(), scaler=self.scaler.state_dict(),
                         state=dict(self.state), board=self.geo.N,
                         cfg=dict(channels=self.args.channels, blocks=self.args.blocks)),
                    os.path.join(out, "latest.pt"))

    def load(self, path, log):
        ck = torch.load(path, map_location=self.dev, weights_only=False)
        self.net.load_state_dict(ck["net"])
        self.tgt.load_state_dict(ck.get("tgt", ck["net"]))
        if "opt" in ck:
            self.opt.load_state_dict(ck["opt"])
        if "scaler" in ck:
            self.scaler.load_state_dict(ck["scaler"])
        self.state.update(ck.get("state", {}))
        log(f"resumed {path}: {self.state['steps']:,} env steps, "
            f"{self.state['updates']:,} updates, {self.state['hours']:.2f} h")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--board", type=int, default=5)
    p.add_argument("--out", default="runs/dqn5")
    p.add_argument("--hours", type=float, default=8.5)
    p.add_argument("--games", type=int, default=256, help="parallel self-play games")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--buffer", type=int, default=400_000)
    p.add_argument("--nstep", type=int, default=3)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=3_000_000)
    p.add_argument("--target-sync", type=int, default=2000)
    p.add_argument("--collect-steps", type=int, default=8, help="env steps between update bursts")
    p.add_argument("--updates", type=int, default=8, help="gradient steps per burst")
    p.add_argument("--min-buffer", type=int, default=50_000)
    p.add_argument("--eval-minutes", type=float, default=20.0)
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--save-minutes", type=float, default=15.0)
    p.add_argument("--resume-from", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-only", action="store_true")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    log = log_to(os.path.join(a.out, "train_log.txt"))
    r = Runner(a)
    latest = os.path.join(a.out, "latest.pt")
    src = a.resume_from and os.path.join(a.resume_from, "latest.pt")
    if os.path.exists(latest):
        r.load(latest, log)
    elif src and os.path.exists(src):
        r.load(src, log)
    rep_path = os.path.join(a.out, "replay.npz")
    if os.path.exists(rep_path):
        r.replay.load(rep_path)
        log(f"replay loaded: {r.replay.n:,}")

    if a.eval_only:
        res = r.eval_vs_bot(a.eval_games, seed=7)
        log(f"EVAL vs heuristic bot: W{res['wins']} D{res['draws']} L{res['losses']} "
            f"score {res['score']:.3f} margin {res['avg_margin']:+.2f}")
        return

    log(f"DQN {a.board}x{a.board} | device {r.dev} | dueling double DQN, n-step {a.nstep}, "
        f"gamma {a.gamma} | {a.games} parallel games | net {a.channels}x{a.blocks} "
        f"({sum(x.numel() for x in r.net.parameters()):,} params) | budget {a.hours:.2f} h")

    t0 = time.time()
    hours_before = r.state["hours"]
    deadline = t0 + a.hours * 3600
    stop = {"v": False}
    try:
        signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))
    except Exception:
        pass
    mpath = os.path.join(a.out, "metrics.csv")
    if not os.path.exists(mpath):
        with open(mpath, "w") as f:
            f.write("hours,env_steps,updates,games,eps,loss,q_mean,target_mean,buffer,vs_bot,vs_bot_margin\n")

    last_save = last_eval = last_log = time.time()
    loss = qm = ym = float("nan")
    best = r.state.get("best_vs_bot", -1.0)
    vs_bot = vs_margin = ""
    while time.time() < deadline and not stop["v"]:
        if os.path.exists(os.path.join(a.out, "STOP")):
            log("stop requested"); break
        r.collect(a.collect_steps)
        if r.replay.n >= a.min_buffer:
            for _ in range(a.updates):
                loss, ym, qm = r.update()
        now = time.time()
        el = hours_before + (now - t0) / 3600
        if now - last_eval > a.eval_minutes * 60 and r.replay.n >= a.min_buffer:
            last_eval = now
            res = r.eval_vs_bot(a.eval_games, seed=r.state["updates"])
            vs_bot, vs_margin = f"{res['score']:.3f}", f"{res['avg_margin']:.2f}"
            tag = ""
            if res["score"] > best:
                best = res["score"]
                r.state["best_vs_bot"] = best
                atomic_save(dict(net=r.net.state_dict(), state=dict(r.state), board=a.board,
                                 cfg=dict(channels=a.channels, blocks=a.blocks), vs_bot=res),
                            os.path.join(a.out, "best.pt"))
                tag = "  -> NEW BEST"
            log(f"EVAL vs heuristic bot: W{res['wins']} D{res['draws']} L{res['losses']} "
                f"score {res['score']:.3f} margin {res['avg_margin']:+.2f}{tag}")
        if now - last_log > 60:
            last_log = now
            log(f"{el:5.2f}h | env steps {r.state['steps']:,} | updates {r.state['updates']:,} | "
                f"games {r.state['games']:,} | eps {r.eps_now():.3f} | loss {loss:.4f} | "
                f"Q {qm:+.3f} target {ym:+.3f} | buffer {r.replay.n:,}")
            with open(mpath, "a") as f:
                f.write(f"{el:.3f},{r.state['steps']},{r.state['updates']},{r.state['games']},"
                        f"{r.eps_now():.4f},{loss:.5f},{qm:.4f},{ym:.4f},{r.replay.n},"
                        f"{vs_bot},{vs_margin}\n")
            vs_bot = vs_margin = ""
            try:
                with open(os.path.join(a.out, "status.json"), "w") as f:
                    json.dump(dict(board=a.board, hours=round(el, 3), env_steps=r.state["steps"],
                                   updates=r.state["updates"], games=r.state["games"],
                                   eps=round(r.eps_now(), 4), loss=loss, best_vs_bot=best,
                                   buffer=r.replay.n, time=time.strftime("%Y-%m-%d %H:%M:%S")), f)
            except OSError:
                pass
        if now - last_save > a.save_minutes * 60:
            last_save = now
            r.state["hours"] = el
            r.save(a.out, "periodic")
            r.replay.save(rep_path)
            log("checkpoint saved")
    r.state["hours"] = hours_before + (time.time() - t0) / 3600
    r.save(a.out, "final")
    r.replay.save(rep_path)
    res = r.eval_vs_bot(a.eval_games, seed=999)
    log(f"FINAL vs heuristic bot: W{res['wins']} D{res['draws']} L{res['losses']} "
        f"score {res['score']:.3f} margin {res['avg_margin']:+.2f}")
    log(f"done: {r.state['hours']:.2f} h, {r.state['games']:,} games, {r.state['updates']:,} updates")


if __name__ == "__main__":
    main()
