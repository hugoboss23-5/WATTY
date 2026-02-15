"""
Watty Benchmark Suite
======================
Not part of CI — run manually:
    python tests/bench.py

Tests recall, store, scan, and cluster at different scales.
Outputs table to stdout + JSON to ~/.watty/benchmark.json.
"""

import sys
import os
import types
import time
import json
import tempfile
import numpy as np

# Mock embedding backend for consistent benchmarks
if "sentence_transformers" not in sys.modules:
    _mock_st = types.ModuleType("sentence_transformers")
    class _MockModel:
        def encode(self, text, **kwargs):
            words = text.lower().split()
            vec = np.zeros(384, dtype=np.float32)
            for w in words:
                np.random.seed(hash(w) % (2**31))
                vec += np.random.randn(384).astype(np.float32)
            norm = np.linalg.norm(vec)
            return (vec / norm) if norm > 0 else vec
    _mock_st.SentenceTransformer = lambda *a, **k: _MockModel()
    sys.modules["sentence_transformers"] = _mock_st

from watty.brain import Brain


def bench_store(brain, count):
    """Benchmark store_memory throughput."""
    start = time.perf_counter()
    for i in range(count):
        brain.store_memory(f"Benchmark memory number {i} about topic {i * 7} with enough words to be realistic")
    elapsed = time.perf_counter() - start
    return {
        "operation": "store_memory",
        "count": count,
        "total_ms": round(elapsed * 1000, 1),
        "per_item_ms": round(elapsed * 1000 / count, 2),
        "throughput": round(count / elapsed, 1),
    }


def bench_recall(brain, queries, label):
    """Benchmark recall latency."""
    # Warm up index
    brain.recall("warmup query")

    times = []
    for q in queries:
        start = time.perf_counter()
        results = brain.recall(q, top_k=10)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return {
        "operation": f"recall ({label})",
        "queries": len(queries),
        "avg_ms": round(sum(times) / len(times), 2),
        "p50_ms": round(sorted(times)[len(times) // 2], 2),
        "p99_ms": round(sorted(times)[int(len(times) * 0.99)], 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
    }


def bench_cluster(brain, label):
    """Benchmark cluster operation."""
    start = time.perf_counter()
    clusters = brain.cluster()
    elapsed = (time.perf_counter() - start) * 1000
    return {
        "operation": f"cluster ({label})",
        "clusters_found": len(clusters),
        "total_ms": round(elapsed, 1),
    }


def bench_scan(brain, tmp, file_count):
    """Benchmark scan_directory throughput."""
    scan_dir = os.path.join(tmp, "bench_scan")
    os.makedirs(scan_dir, exist_ok=True)
    for i in range(file_count):
        with open(os.path.join(scan_dir, f"doc{i}.md"), "w") as f:
            f.write(f"Document {i} about benchmark testing topic {i * 3}\n" * 10)

    start = time.perf_counter()
    result = brain.scan_directory(scan_dir)
    elapsed = (time.perf_counter() - start) * 1000
    return {
        "operation": "scan_directory",
        "files": result["files_scanned"],
        "chunks": result["chunks_stored"],
        "total_ms": round(elapsed, 1),
        "per_file_ms": round(elapsed / max(result["files_scanned"], 1), 2),
    }


def get_memory_mb():
    """Get process RSS in MB."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB
    except Exception:
        return 0


def main():
    results = []
    queries = [
        "machine learning neural networks",
        "distributed systems architecture",
        "Python data science pandas numpy",
        "React frontend hooks components",
        "database optimization indexing",
    ]

    # Scale 1: 100 memories
    print("Benchmarking at 100 memories...")
    tmp = tempfile.mkdtemp()
    brain = Brain(db_path=os.path.join(tmp, "bench.db"))

    r = bench_store(brain, 100)
    results.append(r)
    print(f"  store: {r['throughput']}/s ({r['per_item_ms']}ms each)")

    r = bench_recall(brain, queries * 4, "100 memories")
    results.append(r)
    print(f"  recall: avg={r['avg_ms']}ms p50={r['p50_ms']}ms p99={r['p99_ms']}ms")

    r = bench_cluster(brain, "100 memories")
    results.append(r)
    print(f"  cluster: {r['total_ms']}ms ({r['clusters_found']} clusters)")

    r = bench_scan(brain, tmp, 50)
    results.append(r)
    print(f"  scan: {r['total_ms']}ms for {r['files']} files ({r['per_file_ms']}ms/file)")

    mem_100 = get_memory_mb()
    results.append({"operation": "memory_100", "rss_mb": round(mem_100, 1)})
    print(f"  RAM: {mem_100:.1f}MB")

    # Scale 2: 1k memories
    print("\nBenchmarking at 1k memories...")
    tmp2 = tempfile.mkdtemp()
    brain2 = Brain(db_path=os.path.join(tmp2, "bench.db"))

    r = bench_store(brain2, 1000)
    results.append(r)
    print(f"  store: {r['throughput']}/s ({r['per_item_ms']}ms each)")

    r = bench_recall(brain2, queries * 10, "1k memories")
    results.append(r)
    print(f"  recall: avg={r['avg_ms']}ms p50={r['p50_ms']}ms p99={r['p99_ms']}ms")

    r = bench_cluster(brain2, "1k memories")
    results.append(r)
    print(f"  cluster: {r['total_ms']}ms ({r['clusters_found']} clusters)")

    mem_1k = get_memory_mb()
    results.append({"operation": "memory_1k", "rss_mb": round(mem_1k, 1)})
    print(f"  RAM: {mem_1k:.1f}MB")

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmarks": results,
    }

    try:
        from watty.config import WATTY_HOME
        WATTY_HOME.mkdir(parents=True, exist_ok=True)
        bench_path = WATTY_HOME / "benchmark.json"
        bench_path.write_text(json.dumps(output, indent=2))
        print(f"\nResults saved to {bench_path}")
    except Exception:
        pass

    print("\n" + json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    main()
