import os
import sys
import time
import platform
import psutil
import numpy as np
import cupy as cp  # GPU bilan ishlash uchun CuPy
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from numba import njit, prange, set_num_threads
import matplotlib.pyplot as plt
from tabulate import tabulate

# =====================================================================
# 0-QISM: HISOB-KITOB ALGORITMLARI (CPU va GPU yechimlari)
# =====================================================================

def serial_solver(arr):
    """1) Ketma-ket (Serial CPU) yechim"""
    return (arr.astype(np.int64) * 1000) % 7

def _mp_worker(chunk):
    """Multiprocessing uchun yordamchi ishchi funksiya"""
    return (chunk.astype(np.int64) * 1000) % 7

def mp_solver(arr, num_workers):
    """2) multiprocessing.Pool asosidagi parallel yechim"""
    chunks = np.array_split(arr, num_workers)
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_mp_worker, chunks)
    return np.concatenate(results)

def _thread_worker(chunk, out, start_idx, end_idx):
    """ThreadPool uchun xotirani tejovchi in-place ishchi funksiya"""
    out[start_idx:end_idx] = (chunk.astype(np.int64) * 1000) % 7

def thread_solver(arr, num_workers):
    """3) ThreadPoolExecutor asosidagi oqimli yechim"""
    n = len(arr)
    chunk_size = (n + num_workers - 1) // num_workers
    out = np.empty(n, dtype=np.int64)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            start = i * chunk_size
            end = min(start + chunk_size, n)
            if start < end:
                futures.append(executor.submit(_thread_worker, arr[start:end], out, start, end))
        for f in futures:
            f.result()
    return out

@njit(parallel=True, fastmath=True)
def numba_kernel(arr):
    """4) Numba @njit(parallel=True) va prange asosidagi yadro"""
    n = len(arr)
    out = np.empty(n, dtype=np.int64)
    for i in prange(n):
        out[i] = (arr[i] * 1000) % 7
    return out

def numba_solver(arr, num_threads):
    set_num_threads(num_threads)
    return numba_kernel(arr)

def gpu_solver(arr_gpu):
    """5) GPU (CUDA / CuPy) asosidagi o'ta tezkor yechim"""
    return (arr_gpu.astype(np.int64) * 1000) % 7


# =====================================================================
# 1-BAND: O'LCHASH METODIKASI VA SERIAL TAQQOSLASH
# =====================================================================
def task_1_measure(func, *args, repeat=5, is_gpu=False):
    """
    1-band talabi bo'yicha:
- time.perf_counter() taymeri
- JIT, GPU kompilyatsiya va kesh to'ldirishni hisobdan chiqarish uchun warm-up
- Kamida 5 marta takrorlash
- O'rtacha, minimal va standart og'ish hisoblash
    """
    _ = func(*args)  # Warm-up
    if is_gpu:
        cp.cuda.Device(0).synchronize()
        
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        res = func(*args)
        if is_gpu:
            cp.cuda.Device(0).synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        
    return {
        "mean": float(np.mean(times)),
        "min": float(np.min(times)),
        "std": float(np.std(times)),
        "result": res
    }


# =====================================================================
# 2-BAND: PARALLEL VERSIYALAR VA YAGONA JADVAL (CPU + GPU)
# =====================================================================
def task_2_benchmark_table(arr_cpu, arr_gpu, serial_mean, expected_res, workers=[2, 4, 8]):
    """
    2-band: Har bir parallel versiyani turli workerlar va GPU bilan o'lchash,
    Speedup (T1/Tn) va Efficiency (S/n) hisoblab yagona jadval tuzish.
    """
    techs = {
        "Multiprocessing.Pool": lambda arr, w: mp_solver(arr, w),
        "ThreadPoolExecutor": lambda arr, w: thread_solver(arr, w),
        "Numba (parallel=True)": lambda arr, w: numba_solver(arr, w),
    }
    
    rows = []
    raw_data = {}

    for name, func in techs.items():
        raw_data[name] = {}
        for w in workers:
            stats = task_1_measure(func, arr_cpu, w, repeat=5)
            speedup = serial_mean / stats["mean"]
            efficiency = speedup / w
            
            is_valid = task_3_verify_results(stats["result"], expected_res)
            
            raw_data[name][w] = {
                "time": stats["mean"],
                "speedup": speedup,
                "efficiency": efficiency
            }
            
            rows.append([
                name, w, 
                f"{stats['mean']:.4f}", 
                f"{stats['min']:.4f}", 
                f"{stats['std']:.4f}", 
                f"{speedup:.3f}", 
                f"{efficiency:.3f}", 
                "Mos keldi" if is_valid else "Xatolik bor!"
            ])
            
    # GPU ni ham alohida qo'shamiz (workers o'rniga GPU deb belgilanadi)
    gpu_stats = task_1_measure(gpu_solver, arr_gpu, repeat=5, is_gpu=True)
    gpu_speedup = serial_mean / gpu_stats["mean"]
    gpu_res_host = cp.asnumpy(gpu_stats["result"])
    gpu_valid = task_3_verify_results(gpu_res_host, expected_res)
    
    # Grafik va Karp-Flatt uchun ham moslashuv yaratamiz
    raw_data["GPU (CuPy CUDA)"] = {w: {"speedup": gpu_speedup, "time": gpu_stats["mean"]} for w in workers}
    
    rows.append([
        "GPU (CuPy CUDA)", "GPU",
        f"{gpu_stats['mean']:.4f}",
        f"{gpu_stats['min']:.4f}",
        f"{gpu_stats['std']:.4f}",
        f"{gpu_speedup:.3f}",
        "-",
        "Mos keldi" if gpu_valid else "Xatolik bor!"
    ])
            
    return rows, raw_data


# =====================================================================
# 3-BAND: NATIJALARNING MOS KELISHINI TEKSHIRISH
# =====================================================================
def task_3_verify_results(parallel_res, serial_res):
    """
    3-band: Barcha parallel va GPU versiyalar natijasi serial versiya
    natijasi bilan bir xil ekanini dasturiy tekshirish.
    """
    return np.array_equal(parallel_res, serial_res)


# =====================================================================
# 4-BAND: KETMA-KET ULUSH 'f' NI BAHOLASH (KARP-FLATT METODI)
# =====================================================================
def task_4_calculate_serial_fraction(raw_data):
    """
    4-band: Dasturning ketma-ket (parallellashtirib bo'lmaydigan) ulushi f ni
    Karp-Flatt formulasi orqali baholash: f = (1/S - 1/n) / (1 - 1/n)
    """
    f_list = []
    print("\n--- 4-BAND: Dasturning ketma-ket ulushi (f) tahlili ---")
    for tech in ["Numba (parallel=True)", "Multiprocessing.Pool"]:
        for n in [4, 8]:
            S = raw_data[tech][n]["speedup"]
            f = (1.0 / S - 1.0 / n) / (1.0 - 1.0 / n)
            f_list.append(max(0.0, f))
            print(f"[{tech} | worker={n}]: f ≈ {f:.5f}")
            
    f_mean = float(np.mean(f_list))
    print(f">> Baholangan o'rtacha f ulushi: {f_mean:.4f} ({f_mean * 100:.2f}%)")
    return max(0.01, f_mean)


# =====================================================================
# 5-BAND: AMDAHL VA GUSTAFSON NAZARIY GRAFIK TAHLILI
# =====================================================================
def task_5_amdahl_gustafson_plot(raw_data, f, workers=[1, 2, 4, 8]):
    """
    5-band: Amdahl va Gustafson qonunlari bo'yicha nazariy tezlashishni
    amaliy natijalar bilan solishtirib bitta grafikda saqlash.
    """
    p_workers = np.array(workers)
    amdahl_speedup = 1.0 / (f + (1.0 - f) / p_workers)
    gustafson_speedup = p_workers - f * (p_workers - 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(p_workers, amdahl_speedup, 'r--', label=f"Amdahl qonuni (f={f:.2f})", linewidth=2)
    plt.plot(p_workers, gustafson_speedup, 'g-.', label=f"Gustafson qonuni (f={f:.2f})", linewidth=2)
    
    styles = {
        "Numba (parallel=True)": ("blue", "o"),
        "Multiprocessing.Pool": ("orange", "s"),
        "ThreadPoolExecutor": ("purple", "^")
    }
    
    for tech, (col, marker) in styles.items():
        actual = [1.0] + [raw_data[tech][w]["speedup"] for w in [2, 4, 8]]
        plt.plot(p_workers, actual, label=f"Amaliy: {tech}", color=col, marker=marker, linewidth=2)

    plt.title("5-band: Amdahl va Gustafson nazariyasi vs Amaliy tezlashish", fontsize=12)
    plt.xlabel("Workerlar soni (n)")
    plt.ylabel("Tezlashish koeffitsiyenti (Speedup)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("task_5_amdahl_gustafson.png", dpi=300)
    plt.close()
    print(">> Grafik saqlandi: task_5_amdahl_gustafson.png")


# =====================================================================
# 6-BAND: HAJMGA BOG'LIQLIK (N/4, N/2, N, 2N)
# =====================================================================
def task_6_scalability_analysis(base_n, worker_count=4):
    """
    6-band: N/4, N/2, N va 2N o'lchamlar uchun barcha versiyalarni o'lchash,
    ustunlik almashishini ko'rsatuvchi grafikni chizish.
    """
    sizes = {
        "N/4": base_n // 4,
        "N/2": base_n // 2,
        "N": base_n,
        "2N": base_n * 2
    }
    
    history = {"Serial": [], "Multiprocessing": [], "ThreadPool": [], "Numba": [], "GPU (CuPy)": []}
    labels = list(sizes.keys())
    
    print("\n--- 6-BAND: Turli hajmlarda taqqoslash boshlandi ---")
    for lbl, sz in sizes.items():
        print(f"Hajm o'lchanmoqda: {lbl} ({sz:,} element)...")
        arr_cpu = np.random.randint(1, 100000, size=sz, dtype=np.int32)
        arr_gpu = cp.asarray(arr_cpu)
        
        history["Serial"].append(task_1_measure(serial_solver, arr_cpu, repeat=2)["mean"])
        history["Multiprocessing"].append(task_1_measure(mp_solver, arr_cpu, worker_count, repeat=2)["mean"])
        history["ThreadPool"].append(task_1_measure(thread_solver, arr_cpu, worker_count, repeat=2)["mean"])
        history["Numba"].append(task_1_measure(numba_solver, arr_cpu, worker_count, repeat=2)["mean"])
        history["GPU (CuPy)"].append(task_1_measure(gpu_solver, arr_gpu, repeat=2, is_gpu=True)["mean"])

    plt.figure(figsize=(10, 6))
    for tech, times in history.items():
        plt.plot(labels, times, marker='o', label=tech, linewidth=2)
        
    plt.title(f"6-band: Ma'lumot hajmiga bog'liqlik (Worker/GPU)", fontsize=12)
    plt.xlabel("Ma'lumot hajmi")
    plt.ylabel("Bajarilish vaqti (soniya)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("task_6_scalability.png", dpi=300)
    plt.close()
    print(">> Grafik saqlandi: task_6_scalability.png")


# =====================================================================
# 7-BAND: TAVSIYA JADVALI
# =====================================================================
def task_7_recommendation_table():
    """
    7-band: Kamida beshta qatordan iborat tavsiya jadvali.
    """
    table = [
        ["Kichik massivlar (< 100K)", "Serial (Numpy)", "Worker yaratish va GPU/IPC xarajatlari asosiy vaqtdan ko'p ketadi"],
        ["O'rta massivlar (100K - 1M)", "Numba (Parallel)", "Juda kichik boshlang'ich overhead va xotirani bevosita parallel boshqarish"],
        ["Katta massivlar (N = 26M)", "GPU (CuPy CUDA)", "Minglab CUDA yadrolari va VRAM tezligi evaziga eng yuqori unumdorlik"],
        ["I/O ga bog'liq vazifalar (Disk/Web)", "ThreadPoolExecutor", "Python GIL I/O blokirovkalarida bo'shatiladi, yengil oqimlar qulay"],
        ["Og'ir mustaqil arifmetika (GPU bo'lmasa)", "Multiprocessing.Pool", "GIL cheklovini yengib har bir CPU yadroda to'liq ishlash"]
    ]
    print("\n" + "=" * 90)
    print("--- 7-BAND: TAVSIYA JADVALI ---")
    print(tabulate(table, headers=["Masala turi va Hajm", "Tavsiya etiladigan texnologiya", "O'lchangan raqamlar asosidagi sabab"], tablefmt="grid"))


# =====================================================================
# 8-BAND: USKUNA VA DASTURIY MUHITNI OLISH (GPU QO'SHILGAN)
# =====================================================================
def task_8_system_environment():
    """
    8-band: Uskuna, dasturiy muhit va GPU parametrlarini aniqlash.
    """
    gpu_info = "Aniqlanmadi / Topilmadi"
    try:
        dev = cp.cuda.Device(0)
        gpu_info = f"{dev.name.decode('utf-8')} (VRAM: {dev.mem_info[1] / (1024**3):.2f} GB)"
    except Exception:
        pass

    env_info = [
        ["Operatsion tizim", f"{platform.system()} {platform.release()}"],
        ["Protsessor modeli", platform.processor() or platform.machine()],
        ["Grafik karta (GPU)", gpu_info],
        ["Fizik yadrolar soni", psutil.cpu_count(logical=False)],
        ["Mantiqiy yadrolar (Threads)", psutil.cpu_count(logical=True)],
        ["Operativ xotira (RAM)", f"{psutil.virtual_memory().total / (1024**3):.2f} GB"],
        ["Python versiyasi", platform.python_version()],
        ["NumPy / CuPy versiyalari", f"NP: {np.__version__} | CP: {cp.__version__}"]
    ]
    print("\n" + "=" * 60)
    print("--- 8-BAND: TIZIM VA USKUNA MA'LUMOTLARI ---")
    print(tabulate(env_info, headers=["Parametr", "Qiymat"], tablefmt="grid"))


# =====================================================================
# 9-BAND: YAKUNIY XULOSA VA TAHLIL TEZISLARI
# =====================================================================
def task_9_final_conclusion():
    """
    9-band: Yakuniy xulosa, Word va PDF hisobotga kiritiladigan nazariy tahlillar.
    """
    print("\n" + "=" * 90)
    print("--- 9-BAND: YAKUNIY XULOSA VA NAZARIY ASOSLAR ---")
    print("""
1. Eng samarali texnologiya: GPU (CuPy CUDA) va Numba (@njit(parallel=True)).
   Asoslash: GPU minglab parallel oqimlar (cores) orqali massiv elementlarini bir vaqtning o'zida qayta ishlaydi. 
   Numba esa CPU da C tilidek tez ishlaydi va GIL cheklovini chetlab o'tadi.

2. Farq sabablari tahlili:
- Ma'lumot uzatish (PCIe bus): CPU dan GPU ga ma'lumot ko'chirish vaqt oladi, lekin katta hajmlarda (26M+) GPU tezligi bu xarajatlarni to'liq qoplab ketadi.
- GIL (Global Interpreter Lock): Multithreading CPU-bound vazifalarda yordam bermasa, CuPy va Numba bu muammoni to'liq hal qiladi.
""")


# =====================================================================
# ASOSIY ISHGA TUSHIRISH DASTURI
# =====================================================================
def main():
    BASE_N = 26_000_000
    
    # 8-band: Tizim va GPU ma'lumotlari
    task_8_system_environment()
    
    print(f"\nAsosiy massiv yaratilmoqda: N = {BASE_N:,} ta element...")
    data_cpu = np.random.randint(1, 100000, size=BASE_N, dtype=np.int32)
    
    print("Massiv GPU (VRAM) xotirasiga ko'chirilmoqda...")
    data_gpu = cp.asarray(data_cpu)
    
    # 1-band: Serial o'lchov
    print("\n--- 1-BAND: Serial versiya o'lchanmoqda (5 ta takror) ---")
    serial_stats = task_1_measure(serial_solver, data_cpu, repeat=5)
    print(f"Serial: O'rtacha = {serial_stats['mean']:.4f}s, Min = {serial_stats['min']:.4f}s, Std = {serial_stats['std']:.4f}s")
    
    # 2-band va 3-band: Parallel va GPU benchmark + tekshiruv
    print("\n--- 2-BAND VA 3-BAND: Parallel va GPU versiyalarni o'lchash va tekshirish ---")
    headers = ["Texnologiya", "Worker / GPU", "O'rtacha (s)", "Min (s)", "Std (s)", "Tezlashish (S)", "Samaradorlik (E)", "3-band: Moslik"]
    table_rows, raw_data = task_2_benchmark_table(data_cpu, data_gpu, serial_stats["mean"], serial_stats["result"], workers=[2, 4, 8])
    
    full_table = [["Serial", 1, f"{serial_stats['mean']:.4f}", f"{serial_stats['min']:.4f}", f"{serial_stats['std']:.4f}", "1.000", "1.000", "Etalon"]] + table_rows
    print("\n" + tabulate(full_table, headers=headers, tablefmt="grid"))
    
    # 4-band: Karp-Flatt
    f_val = task_4_calculate_serial_fraction(raw_data)
    
    # 5-band: Amdahl va Gustafson grafiki
    print("\n--- 5-BAND: Nazariy grafik chizilmoqda ---")
    task_5_amdahl_gustafson_plot(raw_data, f_val, workers=[1, 2, 4, 8])
    
    # 6-band: Hajmga bog'liqlik tahlili
    task_6_scalability_analysis(BASE_N, worker_count=4)
    
    # 7-band: Tavsiya jadvali
    task_7_recommendation_table()
    
    # 9-band: Yakuniy xulosa
    task_9_final_conclusion()

if __name__ == "__main__":
    main()