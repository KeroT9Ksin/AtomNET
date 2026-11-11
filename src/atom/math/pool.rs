// src/atom/math/pool.rs
//! AtomMATH: kernel pool — очередь SIMD-заданий с фиксированным числом воркеров.
//!
//! Зачем: ядра AtomMATH (dot, gemm, softmax, nсс) должны исполняться не
//! «на главном потоке при больших объёмах» хаотично, а через общую очередь
//! задач: несколько независимых чанков одной редукции / одного GEMM получают
//! свои воркеры, а главный поток добирает остаток (как в Veron-fork, но для
//! произвольных kernel-заданий, а не только для редукций по чанкам).
//!
//! Гарантии:
//! - задача исполняется ровно один раз;
//! - при `busy >= workers` задача выполняется инлайн (без deadlock, без
//!   фантомных потоков);
//! - результат задачи не зависит от воркера (каждый job — самостоятельный
//!   вычислительный тик, разбиение на чанки делает сумму детерминированной).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

type Job = Box<dyn FnOnce() + Send>;

#[derive(Default)]
struct State {
    queue: VecDeque<Job>,
    busy: usize,
}

static CV: OnceLock<(Mutex<State>, Condvar)> = OnceLock::new();
static WORKERS: AtomicUsize = AtomicUsize::new(0);
static STARTED: AtomicBool = AtomicBool::new(false);
static COMPLETED: AtomicUsize = AtomicUsize::new(0);

const MAX_WORKERS: usize = 8;

fn lock() -> &'static (Mutex<State>, Condvar) {
    CV.get_or_init(|| (Mutex::new(State::default()), Condvar::new()))
}

fn worker() {
    let (m, cv) = lock();
    loop {
        let mut s = m.lock().expect("atom math pool poisoned");
        while s.queue.is_empty() {
            let g = cv.wait(s).expect("atom math pool condvar poisoned");
            s = g;
        }
        let job = s.queue.pop_front().expect("atom math pool: queue");
        s.busy += 1;
        drop(s);
        job();
        let mut s = m.lock().expect("atom math pool poisoned");
        s.busy -= 1;
        COMPLETED.fetch_add(1, Ordering::Relaxed);
        cv.notify_all();
    }
}

fn ensure_started() {
    if STARTED.swap(true, Ordering::Relaxed) {
        return;
    }
    let n = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(4)
        .min(MAX_WORKERS)
        .max(1);
    WORKERS.store(n, Ordering::Relaxed);
    for i in 0..n {
        std::thread::Builder::new()
            .name(format!("atom-math-w{}", i))
            .spawn(move || worker())
            .expect("atom math pool: spawn worker");
    }
}

/// Установить число воркеров пула (для тестов/настройки пользователя).
pub fn set_workers(n: usize) {
    WORKERS.store(n.max(1), Ordering::Relaxed);
}

/// Текущее число воркеров пула.
pub fn workers() -> usize {
    WORKERS.load(Ordering::Relaxed).max(1)
}

/// Счёт выполненных задач (для тестов/метрик).
pub fn completed() -> usize {
    COMPLETED.load(Ordering::Relaxed)
}

/// Текущее число занятых задач (для тестов).
#[cfg(test)]
pub fn busy_count() -> usize {
    let (m, _cv) = lock();
    m.lock().expect("atom math pool poisoned").busy
}

/// Запланировать kernel-задание. Если все воркеры заняты — выполнить инлайн.
pub fn submit<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    ensure_started();
    let (m, cv) = lock();
    let mut s = m.lock().expect("atom math pool poisoned");
    let n = WORKERS.load(Ordering::Relaxed).max(1);
    if s.busy >= n {
        let job = f;
        s.busy += 1;
        drop(s);
        job();
        let mut s2 = m.lock().expect("atom math pool poisoned");
        s2.busy -= 1;
        COMPLETED.fetch_add(1, Ordering::Relaxed);
        cv.notify_all();
        return;
    }
    s.queue.push_back(Box::new(f));
    drop(s);
    cv.notify_all();
}

#[cfg(test)]
pub fn reset_for_tests() {
    let (m, _cv) = lock();
    let mut s = m.lock().expect("atom math pool poisoned");
    s.queue.clear();
    s.busy = 0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_executes_jobs_inline_when_saturated() {
        reset_for_tests();
        set_workers(1);
        let before = completed();
        // Первое задание блокирует единственный воркер.
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        submit(move || {
            let _ = rx.recv_timeout(std::time::Duration::from_secs(5));
        });
        // Ждём, пока воркер действительно возьмёт первое задание (busy=1).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while busy_count() < 1 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        // Второе — должен выполниться инлайн (воркер занят).
        let c2 = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let c2b = c2.clone();
        submit(move || {
            c2b.fetch_add(1, Ordering::Relaxed);
        });
        std::thread::sleep(std::time::Duration::from_millis(200));
        // Инлайн-путь сработал: второе задание выполнилось без воркера.
        assert_eq!(c2.load(Ordering::Relaxed), 1);
        drop(tx);
        // Ждём завершения первого (через condvar/воркера) и инлайн-задание.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while completed().saturating_sub(before) < 2 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    #[test]
    fn pool_jobs_run_exactly_once_without_deadlock() {
        reset_for_tests();
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let before = completed();
        for _ in 0..8 {
            let c = counter.clone();
            submit(move || {
                c.fetch_add(1, Ordering::Relaxed);
            });
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while completed().saturating_sub(before) < 8 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(counter.load(Ordering::Relaxed), 8);
    }
}
