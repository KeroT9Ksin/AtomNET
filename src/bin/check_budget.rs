use atomnet::atom::model::LmConfig;
fn main() {
    for (name, vocab) in [("14B(vocab=151936)", 151936)] {
        let cfg = LmConfig::big(vocab);
        let p = cfg.params();
        let (w, q, train, kv) = cfg.memory_budget_gb();
        println!("{:20} params ≈ {:.1}B: weights fp32 {:.1} GB, Q4 {:.1} GB, train {:.0} GB, kv@1024 {:.1} GB", name, p as f64 / 1e9, w, q, train, kv);
    }
}
