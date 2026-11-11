import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
import atomnet

QL = getattr(atomnet, "PyQairoLivingLayer", None) or atomnet.life.PyQairoLivingLayer


def main():
    np.random.seed(7)
    rows, cols = 4096, 2048
    n_chunks = rows // 256  # 16 чанков

    print(f"Создаём слой {rows}×{cols} ({rows*cols*4/1e6:.1f} MB в f32)")
    W = (np.random.randn(rows, cols) * 0.02).astype(np.float32)
    b = np.zeros(rows, dtype=np.float32)
    layer = QL(W.tolist(), b.tolist())

    # Замер 0: сразу после создания (все чанки в f32 + int4-база)
    mem_0 = layer.arena_memory_bytes() / 1e6
    print(f"\n[0] Сразу после создания: {mem_0:.2f} MB")

    # Фаза 1: греем 8 чанков (0..7), остальные 8..15 остаются холодными
    print("\n=== ФАЗА 1: Нагрев чанков 0..7 ===")
    active_rows = []
    for c in range(8):
        active_rows.extend(range(c * 256, (c + 1) * 256))
    
    grad = np.zeros(rows, dtype=np.float32)
    grad[active_rows] = 1.0
    layer.distribute_grad_np(grad, 1.0)
    
    mem_1_before = layer.arena_memory_bytes() / 1e6
    print(f"Перед epoch_tick: {mem_1_before:.2f} MB")
    
    layer.epoch_tick()  # release_cold освободит чанки 8..15
    
    mem_1_after = layer.arena_memory_bytes() / 1e6
    print(f"После epoch_tick: {mem_1_after:.2f} MB (освобождены чанки 8..15)")
    
    # Фаза 2: устанавливаем бюджет 12 МБ и вызываем rebalance
    print("\n=== ФАЗА 2: Бюджетный губернатор (budget=12 MB) ===")
    layer.set_ram_budget(12 * 1024 * 1024)
    
    mem_2_before = layer.arena_memory_bytes() / 1e6
    print(f"Перед rebalance: {mem_2_before:.2f} MB")
    
    layer.epoch_tick()  # теперь rebalance вытеснит холоднейший из 0..7
    
    mem_2_after = layer.arena_memory_bytes() / 1e6
    print(f"После rebalance: {mem_2_after:.2f} MB (вытеснен 1 чанок)")
    
    # Фаза 3: ужимаем бюджет до 6 МБ
    print("\n=== ФАЗА 3: Агрессивный бюджет (budget=6 MB) ===")
    layer.set_ram_budget(6 * 1024 * 1024)
    layer.epoch_tick()
    
    mem_3 = layer.arena_memory_bytes() / 1e6
    print(f"После rebalance: {mem_3:.2f} MB (вытеснены ещё чанки)")
    
    # Проверка качества: hot err должен остаться 0 для активных
    print("\n=== ПРОВЕРКА КАЧЕСТВА ===")
    hot_errs = []
    for r in active_rows[:32]:
        row_read = np.asarray(layer.arena_read_row_np(r))
        if layer.arena_is_hot(r):
            hot_errs.append(float(np.abs(row_read - W[r]).max()))
    
    if hot_errs:
        print(f"Hot err (активные чанки): {np.mean(hot_errs):.2e} (должно быть ~0)")
    else:
        print("Hot err: нет активных чанков (все вытеснены)")
    
    # Итоговый отчёт
    print("\n=== ИТОГ ===")
    print(f"Начало (все f32)     : {mem_0:.2f} MB")
    print(f"После release_cold   : {mem_1_after:.2f} MB (освобождены 8 холодных)")
    print(f"После budget=12MB    : {mem_2_after:.2f} MB (вытеснен 1 из 8 горячих)")
    print(f"После budget=6MB     : {mem_3:.2f} MB (вытеснены ещё)")
    print(f"Экономия vs f32      : x{mem_0/mem_3:.1f}")
    
    # График
    fig, ax = plt.subplots(figsize=(10, 6))
    phases = ['Создание', 'До release', 'После release', 'Budget=12MB', 'Budget=6MB']
    mems = [mem_0, mem_1_before, mem_1_after, mem_2_after, mem_3]
    
    ax.plot(phases, mems, marker='o', linewidth=2, markersize=8)
    ax.set_ylabel('Memory (MB)', fontsize=12)
    ax.set_title('FlashArena + Region: Memory Management', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    for i, (p, m) in enumerate(zip(phases, mems)):
        ax.annotate(f'{m:.1f} MB', (p, m), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('logs/arena_region_test.png', dpi=150)
    print(f"\nГрафик: logs/arena_region_test.png")


if __name__ == "__main__":
    main()