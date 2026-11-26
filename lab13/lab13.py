from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from random import Random
from statistics import median
from threading import Lock
from time import perf_counter, sleep

BINS = [(0, 59), (60, 69), (70, 79), (80, 89), (90, 100)]


def check_score(x):
    if isinstance(x, bool) or not isinstance(x, int) or not (0 <= x <= 100):
        raise ValueError(f"Некорректный балл: {x}")


def worker_stats(scores):
    count = total = passed = 0
    hist = {f"{a}-{b}": 0 for a, b in BINS}
    for s in scores:
        check_score(s)
        count += 1
        total += s
        if s >= 60:
            passed += 1
        for a, b in BINS:
            if a <= s <= b:
                hist[f"{a}-{b}"] += 1
                break
    return count, total, passed, hist


def worker_io(scores):
    sleep(0.05)
    return worker_stats(scores)


def merge_stats(parts):
    count = sum(p[0] for p in parts)
    total = sum(p[1] for p in parts)
    passed = sum(p[2] for p in parts)
    hist = {f"{a}-{b}": 0 for a, b in BINS}
    for _, _, _, h in parts:
        for k, v in h.items():
            hist[k] += v
    rate = round(passed / count, 4) if count else 0.0
    return count, total, rate, hist


def run_once(groups, fn, pool=None, workers=2):
    if pool is None:
        parts = [fn(g) for g in groups]
    else:
        with pool(max_workers=workers) as ex:
            parts = list(ex.map(fn, groups))
    return merge_stats(parts)


def measure_three(groups, fn, pool=None, workers=2):
    times = []
    res = None
    for _ in range(3):
        t0 = perf_counter()
        res = run_once(groups, fn, pool, workers)
        times.append(perf_counter() - t0)
    return res, median(times), times


def retake_lock_test(limit=5, applicants=20):
    left = limit
    lock = Lock()

    def take_seat(student_id):
        nonlocal left
        with lock:
            if left == 0:
                return student_id, False
            left -= 1
            return student_id, True

    with ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(take_seat, range(applicants)))

    ok = sum(1 for _, granted in res if granted)
    assert ok == limit and left == 0
    return ok, left


def run_tests():
    assert merge_stats([worker_stats([])]) == (
        0, 0, 0.0, {"0-59": 0, "60-69": 0, "70-79": 0, "80-89": 0, "90-100": 0}
    )
    assert merge_stats([worker_stats([60])])[:3] == (1, 60, 1.0)
    assert merge_stats([worker_stats([0, 59, 60, 100])])[:3] == (4, 219, 0.5)
    assert merge_stats([worker_stats([70, 80]), worker_stats([])])[:3] == (2, 150, 1.0)
    try:
        worker_stats([120])
        assert False
    except ValueError:
        pass
    assert retake_lock_test(5, 20) == (5, 0)


def print_report(results, expected, workers, total_items, ok, left, limit, applicants):
    count, total, rate, hist = expected
    line = "=" * 62

    print(f"\n{line}")
    print("  ЛАБОРАТОРНАЯ РАБОТА 13 | ВАРИАНТ 1: СТАТИСТИКА ЭКЗАМЕНА")
    print(line)
    print(f"  Всего оценок: {total_items:,} | Сумма баллов: {total:,}")
    print(f"  Доля сдавших (>= 60): {rate * 100:.2f}% ({rate})")

    print(f"\n{'-' * 62}")
    print("  РАСПРЕДЕЛЕНИЕ ПО ИНТЕРВАЛАМ (ГИСТОГРАММА)")
    print(f"{'-' * 62}")
    for interval, val in hist.items():
        pct = (val / count * 100) if count else 0.0
        bar = "#" * int(pct // 2)
        print(f"  {interval:>6} баллов : {val:>7,} ({pct:5.2f}%)  {bar}")

    print(f"\n{'-' * 62}")
    print(f"  СРАВНЕНИЕ РЕЖИМОВ (workers = {workers}, 3 запуска)")
    print(f"{'-' * 62}")
    print(f"  {'Режим':<24} | {'Медиана':<9} | {'Все 3 замера (с)'}")
    print(f"  {'-' * 24}-+-{'-' * 9}-+-{'-' * 22}")
    for title, med, runs in results:
        runs_str = ", ".join(f"{t:.4f}" for t in runs)
        print(f"  {title:<24} | {med:>7.4f} с | [{runs_str}]")

    print(f"\n{'-' * 62}")
    print("  ПРОВЕРКА КРИТИЧЕСКОЙ СЕКЦИИ (Lock — ЛИМИТ ПЕРЕСДАЧ)")
    print(f"{'-' * 62}")
    print(f"  Заявок подано: {applicants} | Лимит мест: {limit}")
    print(f"  Одобрено: {ok} | Осталось мест: {left} (инвариант соблюден)")
    print(f"{line}\n")


def main():
    run_tests()
    rng = Random(13)
    groups = [[rng.randrange(101) for _ in range(25_000)] for _ in range(8)]
    total_items = sum(len(g) for g in groups)
    workers = 2

    expected = run_once(groups, worker_stats, None, workers)
    modes = [
        ("CPU последовательно", worker_stats, None),
        ("CPU процессы", worker_stats, ProcessPoolExecutor),
        ("I/O последовательно", worker_io, None),
        ("I/O потоки", worker_io, ThreadPoolExecutor),
    ]

    results = []
    for title, fn, pool in modes:
        res, med, runs = measure_three(groups, fn, pool, workers)
        assert res == expected
        results.append((title, med, runs))

    limit, applicants = 5, 20
    ok, left = retake_lock_test(limit, applicants)
    print_report(results, expected, workers, total_items, ok, left, limit, applicants)


if __name__ == "__main__":
    main()