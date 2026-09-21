"""Diverse chess positions labelled with a Stockfish teacher move.

python positions.py [--out runs/positions] [--workers 12] [--train 80000] [--heldout 4000] [--depth 10]
Writes train-NN.jsonl per worker and heldout.jsonl (worker index = --workers, own seed, own games).
"""
import argparse, json, math, multiprocessing as mp, os, random, shutil, time

import chess, chess.engine

STOCKFISH = shutil.which("stockfish") or "stockfish"


def cp_of(score, turn):
    s = score.pov(turn)
    if s.is_mate():
        return 10000 if s.mate() > 0 else -10000
    return s.score()


def worker(wid, target, path, depth, seed, vs_random=False):
    rng = random.Random(seed)
    eng = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    eng.configure({"Threads": 1, "Hash": 16})
    seen, n, g, t0 = set(), 0, 0, time.time()
    with open(path, "a") as f:
        while n < target:
            board = chess.Board()
            for _ in range(rng.randint(0, 8)):
                if board.is_game_over():
                    break
                board.push(rng.choice(list(board.legal_moves)))
            # vs_random: one side is a uniform random mover and only the other side's positions are
            # recorded, so the data holds the won endgames a player must convert against weak play
            weak = rng.choice((chess.WHITE, chess.BLACK)) if vs_random else None
            while not board.is_game_over() and board.ply() < (300 if vs_random else 160) and n < target:
                if board.turn == weak:
                    board.push(rng.choice(list(board.legal_moves)))
                    continue
                key = (board.board_fen(), board.turn)
                if key not in seen:
                    seen.add(key)
                    info = eng.analyse(board, chess.engine.Limit(depth=depth))
                    move = info["pv"][0]
                    if move.promotion in (None, chess.QUEEN):
                        f.write(json.dumps({"fen": board.fen(), "move": move.uci(),
                                            "cp": cp_of(info["score"], board.turn),
                                            "game": f"{wid}-{g}", "ply": board.ply()}) + "\n")
                        f.flush()
                        n += 1
                        if n % 500 == 0:
                            print(f"worker {wid}: {n}/{target} {n / (time.time() - t0):.1f} pos/s", flush=True)
                # noisy low-strength self-play move
                if rng.random() < 0.15:
                    board.push(rng.choice(list(board.legal_moves)))
                    continue
                infos = eng.analyse(board, chess.engine.Limit(depth=6), multipv=4)
                infos = [i for i in infos if "pv" in i]
                cps = [cp_of(i["score"], board.turn) for i in infos]
                best = max(cps)
                w = [math.exp((c - best) / 60) for c in cps]
                board.push(rng.choices(infos, weights=w)[0]["pv"][0])
            g += 1
    eng.quit()
    print(f"worker {wid}: done {n} positions from {g} games", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/positions")
    ap.add_argument("--vs-random", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--train", type=int, default=80000)
    ap.add_argument("--heldout", type=int, default=4000)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(i, -(-a.train // a.workers), f"{a.out}/train-{i:02d}.jsonl", a.depth, a.seed * 1000 + i)
            for i in range(a.workers)]
    jobs.append((a.workers, a.heldout, f"{a.out}/heldout.jsonl", a.depth, a.seed * 1000 + 999))
    t0 = time.time()
    procs = [mp.Process(target=worker, args=(*j, a.vs_random)) for j in jobs]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print(f"all done in {time.time() - t0:.0f}s", flush=True)
