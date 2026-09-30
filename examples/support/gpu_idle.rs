//! Read-only AMD PCI sysfs sampling shared by local acceptance probes.
pub fn sample_idle(
    path: &std::path::Path,
    count: usize,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    if count == 0 || count > 300 {
        return Err("idle samples must be 1..300".into());
    }
    let mut hwmon = std::fs::read_dir(path.join("hwmon"))?;
    let monitor = hwmon.next().ok_or("no GPU hwmon found")??.path();
    let mut busy = 0.0;
    let mut watts = 0.0;
    for _ in 0..count {
        busy += std::fs::read_to_string(path.join("gpu_busy_percent"))?
            .trim()
            .parse::<f64>()?;
        watts += std::fs::read_to_string(monitor.join("power1_average"))?
            .trim()
            .parse::<f64>()?
            / 1e6;
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
    Ok((busy / count as f64, watts / count as f64))
}
