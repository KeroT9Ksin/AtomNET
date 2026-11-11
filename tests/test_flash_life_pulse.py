import os

# Принудительно ограничиваем внешние библиотеки одним потоком.
# Это важно для честного сравнения и диагностики.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import csv
import json
import time

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False

import atomnet


# ═══════════════════════════════════════════════════════════
# Настройки диагностики
# ═══════════════════════════════════════════════════════════

# Небольшая модель для быстрой диагностики.
# Если хочешь тяжелее, поставь:
# NUM_BLOCKS = 8, IN_DIM = 2048, HIDDEN_DIM = 4096
NUM_BLOCKS = 4
IN_DIM = 1024
HIDDEN_DIM = 2048

PULSES = 20
SEED = 42

OUTPUT_CSV_LAYER = "flash_life_layer_records.csv"
OUTPUT_CSV_PULSE = "flash_life_pulse_summary.csv"
OUTPUT_JSON = "flash_life_analysis.json"
OUTPUT_PNG = "flash_life_graph.png"


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def get_flash_layer_class():
    if hasattr(atomnet, "PyFlashLayer"):
        return atomnet.PyFlashLayer

    if hasattr(atomnet, "flash") and hasattr(atomnet.flash, "PyFlashLayer"):
        return atomnet.flash.PyFlashLayer

    raise RuntimeError(
        "PyFlashLayer not found. "
        "Check src/python.rs and FlashLayer registration."
    )


def counts_to_dict(counts):
    # Order from Rust:
    # dead, cold, born, alive, stable, decaying
    return {
        "dead": counts[0],
        "cold": counts[1],
        "born": counts[2],
        "alive": counts[3],
        "stable": counts[4],
        "decaying": counts[5],
    }


def safe_float(x):
    x = float(x)
    if not np.isfinite(x):
        return 0.0
    return x


def build_layers(FlashLayer):
    layers = []

    for _ in range(NUM_BLOCKS):
        # up: IN_DIM -> HIDDEN_DIM
        layers.append(FlashLayer.random(HIDDEN_DIM, IN_DIM, 0.02))

        # down: HIDDEN_DIM -> IN_DIM
        layers.append(FlashLayer.random(IN_DIM, HIDDEN_DIM, 0.02))

    return layers


def run_pulse(layers, x):
    records = []
    current = np.ascontiguousarray(x, dtype=np.float32)

    for i, layer in enumerate(layers):
        input_norm = safe_float(np.linalg.norm(current))

        # Чистый линейный атомарный forward без обновления жизни.
        linear_out = layer.forward_linear_np(current)
        linear_norm = safe_float(np.linalg.norm(linear_out))

        # Живой forward: обновляет Life.
        living_out = layer.forward_living_np(current)
        living_norm = safe_float(np.linalg.norm(living_out))

        counts = layer.state_counts()
        states = counts_to_dict(counts)

        records.append(
            {
                "layer": i,
                "shape": f"{layer.out_features}x{layer.in_features}",
                "input_norm": input_norm,
                "linear_norm": linear_norm,
                "living_norm": living_norm,
                "linear_over_input": safe_float(linear_norm / max(input_norm, 1e-12)),
                "living_over_input": safe_float(living_norm / max(input_norm, 1e-12)),
                **states,
            }
        )

        current = living_out

    return current, records


def analyze(summaries, layer_records):
    analysis = {
        "config": {
            "num_blocks": NUM_BLOCKS,
            "num_layers": NUM_BLOCKS * 2,
            "in_dim": IN_DIM,
            "hidden_dim": HIDDEN_DIM,
            "pulses": PULSES,
            "seed": SEED,
        },
        "issues": [],
        "recommendations": [],
    }

    if not summaries:
        analysis["issues"].append("No summaries collected.")
        return analysis

    final_norms = [s["final_norm"] for s in summaries]
    max_norms = [s["max_norm"] for s in summaries]
    min_norms = [s["min_norm"] for s in summaries]

    mean_final = float(np.mean(final_norms))
    std_final = float(np.std(final_norms))
    cv_final = std_final / max(mean_final, 1e-12)

    analysis["metrics"] = {
        "mean_final_norm": mean_final,
        "std_final_norm": std_final,
        "cv_final_norm": cv_final,
        "last_final_norm": final_norms[-1],
        "last_max_layer_norm": max_norms[-1],
        "last_min_layer_norm": min_norms[-1],
    }

    # Проверяем коллапс.
    collapse_layers = []
    for r in layer_records:
        if r["pulse"] == PULSES - 1 and r["living_norm"] < 1e-6:
            collapse_layers.append(r["layer"])

    if collapse_layers:
        analysis["issues"].append(
            f"Signal collapse detected at layers: {collapse_layers}"
        )
        analysis["recommendations"].append(
            "Add min_gate, energy floor, LayerNorm-like stabilization, or residual pulse."
        )

    # Проверяем взрыв.
    explosion_layers = []
    for r in layer_records:
        if r["pulse"] == PULSES - 1 and r["living_norm"] > 1e3:
            explosion_layers.append(r["layer"])

    if explosion_layers:
        analysis["issues"].append(
            f"Signal explosion detected at layers: {explosion_layers}"
        )
        analysis["recommendations"].append(
            "Add Hexiron dissipation, output clamp, or coherence-based damping."
        )

    # Проверяем рост мёртвых атомов.
    last = summaries[-1]
    total_atoms = (
        last["dead"]
        + last["cold"]
        + last["born"]
        + last["alive"]
        + last["stable"]
        + last["decaying"]
    )

    if total_atoms > 0:
        dead_ratio = last["dead"] / total_atoms
        stable_ratio = last["stable"] / total_atoms

        analysis["metrics"]["dead_ratio_last"] = dead_ratio
        analysis["metrics"]["stable_ratio_last"] = stable_ratio

        if dead_ratio > 0.20:
            analysis["issues"].append(
                f"Too many dead atoms in last pulse: {dead_ratio:.2%}"
            )
            analysis["recommendations"].append(
                "Life is too aggressive: raise energy_min tolerance, add low-energy patience, add min_gate."
            )

        if stable_ratio < 0.05 and PULSES >= 10:
            analysis["issues"].append(
                f"Very few stable atoms after {PULSES} pulses: {stable_ratio:.2%}"
            )
            analysis["recommendations"].append(
                "Life does not stabilize: lower stability threshold or increase coherence signal."
            )

    # Ровность пульса.
    if cv_final > 0.5:
        analysis["issues"].append(
            f"Pulse is unstable: CV(final_norm)={cv_final:.3f}"
        )
        analysis["recommendations"].append(
            "Add pulse governor: target norm tracking and soft gain correction."
        )
    else:
        analysis["recommendations"].append(
            "Pulse is reasonably stable; can proceed to Qairo v2 optimization."
        )

    return analysis


def save_csv_layer(records):
    if not records:
        return

    keys = list(records[0].keys())

    with open(OUTPUT_CSV_LAYER, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def save_csv_pulse(summaries):
    if not summaries:
        return

    keys = list(summaries[0].keys())

    with open(OUTPUT_CSV_PULSE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)


def save_plot(summaries, layer_records):
    if not HAS_PLOT:
        print("matplotlib not found; skipping PNG graph.")
        return

    if not summaries or not layer_records:
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    # 1) Нормы по слоям для первого и последнего пульса.
    ax = axes[0]

    first = [r for r in layer_records if r["pulse"] == 0]
    last = [r for r in layer_records if r["pulse"] == PULSES - 1]

    if first:
        xs = [r["layer"] for r in first]
        ax.plot(
            xs,
            [max(r["living_norm"], 1e-12) for r in first],
            marker="o",
            label="pulse 0 living_norm",
        )

    if last:
        xs = [r["layer"] for r in last]
        ax.plot(
            xs,
            [max(r["living_norm"], 1e-12) for r in last],
            marker="o",
            label=f"pulse {PULSES - 1} living_norm",
        )

    ax.set_yscale("log")
    ax.set_xlabel("layer index")
    ax.set_ylabel("living_norm (log)")
    ax.set_title("FLASH/Life pulse: layer norms")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # 2) Финальная норма по пульсам.
    ax = axes[1]
    ax.plot(
        [s["pulse"] for s in summaries],
        [s["final_norm"] for s in summaries],
        marker="o",
        label="final_norm",
    )
    ax.set_xlabel("pulse")
    ax.set_ylabel("final_norm")
    ax.set_title("Pulse stabilization")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3) Life states over pulses.
    ax = axes[2]

    pulses = [s["pulse"] for s in summaries]
    dead = [s["dead"] for s in summaries]
    cold = [s["cold"] for s in summaries]
    born = [s["born"] for s in summaries]
    alive = [s["alive"] for s in summaries]
    stable = [s["stable"] for s in summaries]
    decaying = [s["decaying"] for s in summaries]

    ax.stackplot(
        pulses,
        dead,
        cold,
        born,
        alive,
        stable,
        decaying,
        labels=[
            "dead",
            "cold",
            "born",
            "alive",
            "stable",
            "decaying",
        ],
    )

    ax.set_xlabel("pulse")
    ax.set_ylabel("atoms")
    ax.set_title("Life state evolution")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=140)
    print(f"Graph saved: {OUTPUT_PNG}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    print("=== FLASH + Life Pulse Diagnostic ===")
    print(f"matplotlib: {'yes' if HAS_PLOT else 'no'}")

    FlashLayer = get_flash_layer_class()

    np.random.seed(SEED)

    print("\n[1] Building layers...")

    layers = build_layers(FlashLayer)

    total_params = sum(
        layer.out_features * layer.in_features for layer in layers
    )

    print(f"layers: {len(layers)}")
    print(f"params: {total_params / 1e6:.2f} M")

    x0 = (np.random.randn(IN_DIM) * 0.02).astype(np.float32)

    summaries = []
    all_layer_records = []

    print("\n[2] Running pulses...")

    for pulse in range(PULSES):
        t0 = time.perf_counter()

        final, records = run_pulse(layers, x0)

        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Add pulse id to layer records.
        for r in records:
            r["pulse"] = pulse

        all_layer_records.extend(records)

        # Aggregate states.
        agg = {
            "dead": 0,
            "cold": 0,
            "born": 0,
            "alive": 0,
            "stable": 0,
            "decaying": 0,
        }

        living_norms = []

        for r in records:
            living_norms.append(r["living_norm"])
            for k in agg.keys():
                agg[k] += r[k]

        final_norm = safe_float(np.linalg.norm(final))

        summary = {
            "pulse": pulse,
            "time_ms": dt_ms,
            "final_norm": final_norm,
            "max_norm": safe_float(max(living_norms)),
            "min_norm": safe_float(min(living_norms)),
            "mean_norm": safe_float(np.mean(living_norms)),
            "std_norm": safe_float(np.std(living_norms)),
            **agg,
        }

        summaries.append(summary)

        print(
            f"pulse {pulse:02d} | "
            f"time {dt_ms:8.3f} ms | "
            f"final {final_norm:10.6f} | "
            f"max {summary['max_norm']:10.6f} | "
            f"min {summary['min_norm']:10.6f} | "
            f"dead {agg['dead']:6d} | "
            f"alive {agg['alive']:6d} | "
            f"stable {agg['stable']:6d}"
        )

    print("\n[3] Saving reports...")

    save_csv_layer(all_layer_records)
    save_csv_pulse(summaries)

    print(f"Layer CSV: {OUTPUT_CSV_LAYER}")
    print(f"Pulse CSV: {OUTPUT_CSV_PULSE}")

    analysis = analyze(summaries, all_layer_records)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"Analysis JSON: {OUTPUT_JSON}")

    save_plot(summaries, all_layer_records)

    print("\n[4] Analysis ===")

    if "metrics" in analysis:
        for k, v in analysis["metrics"].items():
            print(f"{k}: {v}")

    if analysis["issues"]:
        print("\nIssues:")
        for issue in analysis["issues"]:
            print(f"  - {issue}")

    if analysis["recommendations"]:
        print("\nRecommendations:")
        for rec in analysis["recommendations"]:
            print(f"  - {rec}")

    print("\nFLASH LIFE PULSE DIAGNOSTIC FINISHED")


if __name__ == "__main__":
    main()