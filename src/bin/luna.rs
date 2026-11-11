// src/bin/luna.rs
//! Бинарь-обвязка AtomNET: математика, саморазвитие, text-LM.
//!
//! Строгая валидация аргументов: опечатка/неверное значение ->
//! понятная ошибка на stderr + exit code 2 (никакой «тихой» подстановки
//! значений по умолчанию для нечисловых аргументов).
//!
//! Использование:
//!   cargo run --release --bin luna -- 260 700          # curriculum, авто-потоки
//!   cargo run --release --bin luna -- 260 700 8        # threads=8
//!   cargo run --release --bin luna -- compare 100 400  # саморазвитие vs датасет
//!   cargo run --release --bin luna -- text <file> <epochs> <threads> <temp> <d> <blocks> <window>
//!   cargo run --release --bin luna -- text data/corpus.txt 200 6 0.7 96 2 128

use std::env;

const TEXT_USAGE: &str = "luna text <file> <epochs> <threads> <temp> <d> <blocks> <window>";

fn usage_error(msg: &str) -> ! {
    eprintln!("[luna] ОШИБКА: {msg}");
    eprintln!("[luna] использование: {TEXT_USAGE}");
    std::process::exit(2);
}

/// Строгий разбор позиционного числового аргумента: None если аргумента
/// нет (тогда вызывающий код берёт дефолт), ошибка если не число.
fn arg<T: std::str::FromStr>(args: &[String], idx: usize, name: &str) -> Option<T> {
    match args.get(idx) {
        None => None,
        Some(s) => match s.parse::<T>() {
            Ok(v) => Some(v),
            Err(_) => usage_error(&format!("{name} = '{s}': ожидалось число")),
        },
    }
}

/// Требование с понятным сообщением.
fn require(cond: bool, msg: &str) {
    if !cond {
        usage_error(msg);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let has_text = args.iter().any(|s| s == "text");
    let has_compare = args.iter().any(|s| s == "compare");

    // Потоки: флаг --threads N, иначе позиционный (4 для text/compare, 3
    // иначе), иначе авто по физическим ядрам.
    let threads: usize = args
        .iter()
        .position(|s| s == "--threads")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .or_else(|| {
            let idx = if has_text || has_compare { 4 } else { 3 };
            arg::<usize>(&args, idx, "threads")
        })
        .filter(|&t| t >= 1)
        .unwrap_or_else(atomnet::atom::flash::flash_hw::recommend_threads);
    atomnet::atom::prod::setup_threads(threads);

    if has_compare {
        let e1 = arg::<usize>(&args, 2, "stage1").unwrap_or(100);
        let e2 = arg::<usize>(&args, 3, "stage2").unwrap_or(400);
        require(e1 >= 1, &format!("stage1 = {e1}: должен быть >= 1"));
        require(e2 >= 1, &format!("stage2 = {e2}: должен быть >= 1"));
        println!("[luna] compare mode: stage1={e1}, stage2={e2}");
        atomnet::atom::prod::run_compare(e1, e2);
        return;
    }

    if has_text {
        let file = args.get(2).cloned().unwrap_or_else(|| "corpus.txt".into());
        let epochs = arg::<usize>(&args, 3, "epochs").unwrap_or(200);
        let temp = arg::<f32>(&args, 5, "temp").unwrap_or(0.7);
        let d = arg::<usize>(&args, 6, "d").unwrap_or(96);
        let blocks = arg::<usize>(&args, 7, "blocks").unwrap_or(2);
        let window = arg::<usize>(&args, 8, "window").unwrap_or(128);

        // ── Строгая валидация (text-LM) ──
        require(epochs >= 1, &format!("epochs = {epochs}: должен быть >= 1"));
        require(
            temp > 0.0 && temp <= 2.0,
            &format!("temp = {temp}: должен быть в интервале (0, 2]"),
        );
        require(
            d >= 16 && d % 8 == 0,
            &format!("d = {d}: должен быть >= 16 и кратен 8 (d_head = d/2)"),
        );
        require(blocks >= 1, &format!("blocks = {blocks}: должен быть >= 1"));
        require(window >= 16, &format!("window = {window}: должен быть >= 16"));
        require(
            std::path::Path::new(&file).exists(),
            &format!("файл корпуса '{file}' не найден"),
        );

        println!(
            "[luna] text mode: file={file}, epochs={epochs}, threads={threads}, temp={temp}, d={d}, blocks={blocks}, window={window}"
        );
        // Лимит корпуса: env ATOMNET_TEXT_MAX_BYTES (по умолчанию 200 000).
        let max_bytes: usize = std::env::var("ATOMNET_TEXT_MAX_BYTES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(200_000);
        require(
            max_bytes >= 1024,
            &format!("ATOMNET_TEXT_MAX_BYTES = {max_bytes}: должен быть >= 1024"),
        );
        println!("[luna] max_bytes={max_bytes}");
        atomnet::atom::prod::run_text_lm(&file, epochs, max_bytes, temp, d, blocks, window);
        return;
    }

    // Математика: luna <stage1> <stage2> [--save|--resume path]
    let e1 = arg::<usize>(&args, 1, "stage1").unwrap_or(100);
    let e2 = arg::<usize>(&args, 2, "stage2").unwrap_or(300);
    require(e1 >= 1, &format!("stage1 = {e1}: должен быть >= 1"));
    require(e2 >= 1, &format!("stage2 = {e2}: должен быть >= 1"));
    let ckpt = args
        .iter()
        .position(|a| a == "--save" || a == "--resume")
        .and_then(|i| args.get(i + 1))
        .cloned();
    println!(
        "[luna] AtomNET math curriculum: stage1={e1} epochs, stage2={e2} epochs, ckpt={ckpt:?}"
    );
    atomnet::atom::prod::run_math_curriculum(e1, e2, ckpt.as_deref());
}
