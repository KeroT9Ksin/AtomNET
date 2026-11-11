use atomnet::atom::math::binjit as b;
fn main() {
    let a = vec![0xAAu8; 8];
    let b = vec![0x55u8; 8];
    let c = b::xor_jit(&a, &b);
    println!("n=8: {:?}", c);
    let a = vec![0xAAu8; 256];
    let b = vec![0x55u8; 256];
    let c = b::xor_jit(&a, &b);
    let ok = c.iter().all(|v| *v == 0xFF);
    println!("n=256 ok={}", ok);
}
