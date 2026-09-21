fn main() -> candle_core::Result<()> {
    let path = std::env::args().nth(1).expect("GGUF path required");
    let content =
        candle_core::quantized::gguf_file::Content::read(&mut std::fs::File::open(path)?)?;
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    let mut maximum = (0usize, String::new());
    for (name, info) in &content.tensor_infos {
        if std::env::args().any(|a|a=="--shapes") && (name.starts_with("blk.0.")||name=="output.weight") { println!("{name} {:?} {:?}",info.ggml_dtype,info.shape); }
        *counts.entry(format!("{:?}", info.ggml_dtype)).or_default() += 1;
        let bytes =
            info.shape.elem_count() / info.ggml_dtype.block_size() * info.ggml_dtype.type_size();
        if bytes > maximum.0 {
            maximum = (bytes, name.clone());
        }
    }
    println!("dtype_counts={counts:?}\nmaximum_packed_tensor={maximum:?}");
    Ok(())
}
