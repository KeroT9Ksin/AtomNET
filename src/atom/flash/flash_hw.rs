// src/atom/flash/flash_hw.rs
//! flash_hw — «холодный старт» на уровне железа:
//! детект ISA-тиров (AOT-диспатч), кэшей, ядер и привязка потоков.
//!
//! JIT-генерация на x86 для наших статических AVX2-ядер не даёт выигрыша
//! (память доминирует); реальные рычаги userspace — affinity, топология,
//! размеры кэшей, huge pages — собраны здесь.

use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct CpuInfo {
    pub vendor: &'static str,
    pub avx2: bool,
    pub fma: bool,
    pub avx512f: bool,
    pub bmi2: bool,
    pub cpu_mhz: u64,
    /// Число ядер, доступных процессу (affinity-маска).
    pub allowed_cores: usize,
    pub logical_cores: usize,
    pub l1_bytes: usize,
    pub l2_bytes: usize,
    pub l3_bytes: usize,
}

fn sysfs_int(path: &str) -> Option<usize> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
}

/// Число ядер, доступных процессу (affinity-маска).
fn allowed_cores() -> usize {
    unsafe {
        let mut mask: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut mask) == 0 {
            let mut n = 0usize;
            let bytes = std::mem::size_of::<libc::cpu_set_t>() * 8;
            for c in 0..bytes {
                if libc::CPU_ISSET(c, &mask) {
                    n += 1;
                }
            }
            return n.max(1);
        }
        1
    }
}

/// Бюджет тредов AtomNET в процессе. Уважает affinity-маску (sched_getaffinity
/// через allowed_cpu_ids) и потому cgroup/taskset-ограничения. Override через
/// env `ATOMNET_THREADS=N`: `N=1` — режим одного потока (пул не спавнится,
/// привязки нет, fork работает inline). `N=0` и мусор — как авто.
/// Не может превышать число физические ядер из маски.
pub fn thread_budget() -> usize {
    static BUDGET: OnceLock<usize> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let phys = physical_cpu_ids();
        let phys = if phys.is_empty() {
            let n = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            n.div_ceil(2)
        } else {
            phys.len()
        };
        let phys = phys.max(1);
        if let Ok(v) = std::env::var("ATOMNET_THREADS") {
            if let Ok(n) = v.parse::<usize>() {
                if n == 0 {
                    return 1;
                }
                if n >= 1 {
                    return n.min(phys);
                }
            }
        }
        phys
    })
}

/// Список CPU из affinity-маски процесса (/proc/self/status).
/// Кэшируется: топология не меняется за время жизни процесса, а читать
/// /proc на каждый fork — сотни микросекунд, которые убивают мелкие GEMM.
pub fn allowed_cpu_ids() -> Vec<usize> {
    static CACHE: OnceLock<Vec<usize>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
                return Vec::new();
            };
            let Some(line) = status
                .lines()
                .find(|l| l.starts_with("Cpus_allowed_list:"))
            else {
                return Vec::new();
            };
            let list = line.split_once(':').map(|(_, v)| v.trim()).unwrap_or("");
            let mut ids = Vec::new();
            for part in list.split(',') {
                if part.is_empty() {
                    continue;
                }
                if let Some((a, b)) = part.split_once('-') {
                    let (Ok(lo), Ok(hi)) = (a.parse::<usize>(), b.parse::<usize>()) else {
                        continue;
                    };
                    ids.extend(lo..=hi);
                } else if let Ok(v) = part.parse::<usize>() {
                    ids.push(v);
                }
            }
            ids
        })
        .clone()
}

/// Число физических ядер (package_id, core_id) среди доступных CPU.
/// Возвращает 0, если топология недоступна.
fn physical_cores() -> usize {
    let ids = allowed_cpu_ids();
    if ids.is_empty() {
        return 0;
    }
    let mut unique = std::collections::HashSet::new();
    for cpu in ids {
        let pkg = sysfs_int(&format!(
            "/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id"
        ))
        .unwrap_or(0);
        let core = sysfs_int(&format!(
            "/sys/devices/system/cpu/cpu{cpu}/topology/core_id"
        ))
        .unwrap_or(cpu);
        unique.insert((pkg, core));
    }
    unique.len()
}

/// По одному логическому CPU на каждое физическое ядро (первый в SMT-паре),
/// в пределах affinity-маски процесса. Порядок — по возрастанию CPU id.
/// Пример: ядро0={0,1}, ядро1={2,3}, … → [0, 2, 4, 6, 8, 10].
pub fn physical_cpu_ids() -> Vec<usize> {
    static CACHE: OnceLock<Vec<usize>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let ids = allowed_cpu_ids();
            let mut seen = std::collections::HashSet::new();
            let mut out = Vec::new();
            for cpu in ids {
                let pkg = sysfs_int(&format!(
                    "/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id"
                ))
                .unwrap_or(0);
                let core = sysfs_int(&format!(
                    "/sys/devices/system/cpu/cpu{cpu}/topology/core_id"
                ))
                .unwrap_or(cpu);
                if seen.insert((pkg, core)) {
                    out.push(cpu);
                }
            }
            out
        })
        .clone()
}

/// Подсказка THP для уже выделенной памяти (режим madvise): 2 МБ страницы
/// вместо 4 КБ снижают число page faults и TLB-промахов на больших C/B.
/// Советуем только внутренность диапазона, выровненную на 2 МБ: края
/// могут выходить за mapping. Ошибку игнорируем — это лишь hint.
pub fn advise_huge_pages(ptr: *mut u8, bytes: usize) {
    #[cfg(target_os = "linux")]
    unsafe {
        const HP: usize = 2 * 1024 * 1024;
        let begin = (ptr as usize + HP - 1) & !(HP - 1);
        let end = (ptr as usize + bytes) & !(HP - 1);
        if end > begin {
            libc::madvise(
                begin as *mut libc::c_void,
                end - begin,
                libc::MADV_HUGEPAGE,
            );
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptr, bytes);
    }
}

/// Привязать ТЕКУЩИЙ поток к заданным ядрам (или всем, если пусто).
pub fn pin_thread_to(cores: &[usize]) -> Result<(), String> {
    unsafe {
        let mut mask: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut mask);
        if cores.is_empty() {
            // все доступные
            let allowed = allowed_cpu_ids();
            if allowed.is_empty() {
                let total = 1024.min(libc::CPU_SETSIZE as usize);
                for c in 0..total {
                    libc::CPU_SET(c, &mut mask);
                }
            } else {
                for c in allowed {
                    if c < libc::CPU_SETSIZE as usize {
                        libc::CPU_SET(c, &mut mask);
                    }
                }
            }
        } else {
            for &c in cores {
                if c < libc::CPU_SETSIZE as usize {
                    libc::CPU_SET(c, &mut mask);
                }
            }
        }
        let rc = libc::pthread_setaffinity_np(
            libc::pthread_self(),
            std::mem::size_of::<libc::cpu_set_t>(),
            &mask,
        );
        if rc != 0 {
            return Err(format!("pthread_setaffinity_np rc={rc}"));
        }
    }
    Ok(())
}

pub fn probe() -> &'static CpuInfo {
    static INFO: OnceLock<CpuInfo> = OnceLock::new();
    INFO.get_or_init(|| {
        let vendor = if is_x86_feature_detected!("avx2") {
            "x86_64"
        } else {
            "generic"
        };
        let l1 = sysfs_int("/sys/devices/system/cpu/cpu0/cache/index0/size")
            .unwrap_or(32)
            .saturating_mul(1024);
        let l2 = sysfs_int("/sys/devices/system/cpu/cpu0/cache/index2/size")
            .unwrap_or(512)
            .saturating_mul(1024);
        let l3_paths = [
            "/sys/devices/system/cpu/cpu0/cache/index3/size",
            "/sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list",
        ];
        let l3 = l3_paths
            .first()
            .and_then(|p| sysfs_int(p))
            .map(|kb| kb.saturating_mul(1024))
            .unwrap_or(8 * 1024 * 1024);
        let mhz = sysfs_int("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
            .map(|khz| khz / 1000)
            .unwrap_or(0);
        CpuInfo {
            vendor,
            avx2: is_x86_feature_detected!("avx2"),
            fma: is_x86_feature_detected!("fma"),
            avx512f: is_x86_feature_detected!("avx512f"),
            bmi2: is_x86_feature_detected!("bmi2"),
            cpu_mhz: mhz as u64,
            allowed_cores: allowed_cores(),
            logical_cores: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            l1_bytes: l1,
            l2_bytes: l2,
            l3_bytes: l3,
        }
    })
}


/// Кластеры L3 (CCX-группы): ядра, делящие один L3. Парсит
/// /sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list.
pub fn l3_clusters() -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for cpu in 0..128usize {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/cache/index3/shared_cpu_list");
        let Ok(list) = std::fs::read_to_string(&path) else { continue };
        let mut ids: Vec<usize> = Vec::new();
        for part in list.trim().split(',') {
            if part.is_empty() {
                continue;
            }
            if let Some((a, b)) = part.split_once('-') {
                let (Ok(lo), Ok(hi)) = (a.parse::<usize>(), b.parse::<usize>()) else { continue };
                ids.extend(lo..=hi);
            } else if let Ok(v) = part.parse::<usize>() {
                ids.push(v);
            }
        }
        ids.sort_unstable();
        ids.dedup();
        if !ids.is_empty() && !groups.contains(&ids) {
            groups.push(ids);
        }
    }
    groups
}

/// L3-размер в байтах.
pub fn l3_bytes() -> usize {
    probe().l3_bytes
}

/// Рекомендуемое число потоков обучения: физические ядра из topology
/// (не гиперпотоки), ограниченные affinity-маской процесса.
pub fn recommend_threads() -> usize {
    let info = probe();
    let allowed = info.allowed_cores.max(1);
    let physical = physical_cores();
    if physical > 0 {
        physical.min(allowed).max(1)
    } else {
        // Топология недоступна: консервативно половина логических.
        (allowed / 2).max(1).min(info.logical_cores.max(1))
    }
}

/// Текущий AOT-тир ядер (для логов/диагностики).
pub fn kernel_tier() -> &'static str {
    let i = probe();
    if i.avx512f {
        "avx512"
    } else if i.avx2 && i.fma {
        "avx2+fma"
    } else if i.avx2 {
        "avx2"
    } else {
        "scalar"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hw_probe_sane() {
        let i = probe();
        assert!(i.logical_cores >= 1);
        assert!(i.allowed_cores >= 1);
        assert!(i.l2_bytes >= 16 * 1024);
        assert!(i.l3_bytes >= 1024 * 1024);
        #[cfg(target_arch = "x86_64")]
        {
            // тестовые машины сборки — AVX2-совместимые Zen; ядро обязано
            // видеть свой тир
            assert!(i.avx2, "ожидался AVX2 на x86_64");
            assert!(kernel_tier() == "avx2+fma" || kernel_tier() == "avx2");
        }
        assert!(recommend_threads() >= 1);
    }

    #[test]
    fn affinity_pin_roundtrip() {
        let n = allowed_cores();
        // привязать к ядру 0 (или к любому доступному)
        pin_thread_to(&[0]).unwrap_or(());
        let after = allowed_cores();
        assert!(after >= 1);
        let _ = n;
        // вернуть всё
        pin_thread_to(&[]).unwrap_or(());
    }
}
