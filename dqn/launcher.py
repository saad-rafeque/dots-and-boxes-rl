# ==== DOTS & BOXES - DQN training launcher (Kaggle, GPU T4 x2) ====
# Two DQN runs side by side, one per GPU, same code, different hyper-parameters.
# A: steady   - slow exploration decay, small steps, 3-step returns
# B: aggressive - fast decay, bigger steps, 5-step returns, more updates
# Both stream their logs here, checkpoint every 15 min, and stop before the budget.
import os, sys, time, json, glob, shutil, signal, threading, subprocess

TOTAL_HOURS = float(os.environ.get("DQN_HOURS", 8.5))
WORK = os.environ.get("DQN_WORK", "/kaggle/working")
BOARD = 5
T0 = time.time()
os.chdir(WORK)


def say(m):
    print(time.strftime("%H:%M:%S ") + "[launcher] " + m, flush=True)


try:
    import numba  # noqa
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numba"], check=True)
import torch, numba

ngpu = torch.cuda.device_count()
mem_gb = int(open("/proc/meminfo").readline().split()[1]) / 1e6
say(f"python {sys.version.split()[0]} | torch {torch.__version__} | numba {numba.__version__} | "
    f"cpus {os.cpu_count()} | RAM {mem_gb:.1f} GB | "
    f"GPUs {[torch.cuda.get_device_name(i) for i in range(ngpu)]}")
if ngpu == 0:
    say("WARNING: no GPU - turn on 'GPU T4 x2' in Settings > Accelerator.")

# --- sanity check the game core before spending 8 hours on it -----------------
chk = subprocess.run([sys.executable, "selftest.py"], capture_output=True, text=True)
print(chk.stdout[-4000:], chk.stderr[-3000:], flush=True)
if chk.returncode != 0:
    raise SystemExit("self-test failed - not starting training")

# --- resume from a previous notebook's output, if one is attached -------------
def find_resume(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/runs/{name}/latest.pt", recursive=True))
    return os.path.dirname(hits[-1]) if hits else None

hours = TOTAL_HOURS - (time.time() - T0) / 3600 - 0.05
games = 384 if mem_gb >= 20 else 192

RUNS = [
    ("dqn_a", ["--lr", "3e-4", "--nstep", "3", "--eps-decay-steps", "4000000",
               "--target-sync", "2000", "--updates", "8", "--collect-steps", "8"]),
    ("dqn_b", ["--lr", "1e-3", "--nstep", "5", "--eps-decay-steps", "1500000",
               "--target-sync", "1000", "--updates", "12", "--collect-steps", "8"]),
]
say(f"starting {len(RUNS)} DQN runs, {hours:.2f} h each, {games} parallel games, board {BOARD}x{BOARD}")

procs = []
for i, (name, extra) in enumerate(RUNS):
    env = dict(os.environ, PYTHONUNBUFFERED="1", NUMBA_CACHE_DIR=f"/tmp/numba_{name}")
    if ngpu:
        env["CUDA_VISIBLE_DEVICES"] = str(i % ngpu)
    cmd = [sys.executable, "-u", "train_dqn.py", "--board", str(BOARD), "--out", f"runs/{name}",
           "--hours", f"{hours:.3f}", "--games", str(games), "--channels", "64", "--blocks", "6",
           "--batch", "512", "--buffer", "600000", "--min-buffer", "50000",
           "--eval-minutes", "20", "--eval-games", "200", "--save-minutes", "15",
           "--seed", str(1234 + i)] + extra
    res = find_resume(name)
    if res:
        cmd += ["--resume-from", res]
        say(f"{name}: resuming from {res}")
    say(f"{name}: " + " ".join(cmd[2:]))
    procs.append((name, subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1)))


def relay(name, p):
    for line in p.stdout:
        print(f"[{name}] {line}", end="", flush=True)


threads = [threading.Thread(target=relay, args=x, daemon=True) for x in procs]
for t in threads:
    t.start()

last_beat = time.time()
sent = False
while any(p.poll() is None for _, p in procs):
    el = (time.time() - T0) / 3600
    if el > TOTAL_HOURS + 0.05 and not sent:
        say("budget reached - asking both runs to save and stop")
        for _, p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        sent = True
    if el > TOTAL_HOURS + 0.2:
        say("hard stop")
        for _, p in procs:
            if p.poll() is None:
                p.kill()
    if time.time() - last_beat > 900:
        last_beat = time.time()
        try:
            smi = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                                  "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        except Exception:
            smi = "n/a"
        say(f"heartbeat {el:.2f} h | GPU: {smi.replace(chr(10), ' ; ')}")
    time.sleep(20)

for t in threads:
    t.join(timeout=10)
say("return codes: " + str({n: p.returncode for n, p in procs}))

os.makedirs("models", exist_ok=True)
for name, _ in RUNS:
    for f in ("best.pt", "latest.pt"):
        src = f"runs/{name}/{f}"
        if os.path.exists(src):
            shutil.copy2(src, f"models/{name}_{f}")
    st = f"runs/{name}/status.json"
    if os.path.exists(st):
        say(f"{name} final status: " + open(st).read())
say(f"all done in {(time.time() - T0) / 3600:.2f} h. Download models/dqn_a_best.pt and models/dqn_b_best.pt")
