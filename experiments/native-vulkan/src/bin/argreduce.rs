use candle_core::{Device, Result, Tensor};
use std::time::Instant;

fn oracle(row: &[f32], maximum: bool) -> u32 {
    let mut best = 0;
    for i in 1..row.len() {
        if if maximum {
            row[i] > row[best]
        } else {
            row[i] < row[best]
        } {
            best = i;
        }
    }
    best as u32
}

fn main() -> Result<()> {
    if std::env::args().any(|a| a == "--validate-shader") {
        let source = include_str!("../../../../vendor/candle-core/src/vulkan_arg_reduce.wgsl")
            .replace("__HALF_LANES__", "128")
            .replace("__LANES__", "256")
            .replace("__ROWS__", "6")
            .replace("__WIDTH__", "248320")
            .replace("__OFFSET__", "0")
            .replace("__COMPARE__", ">");
        let source =
            format!("fn dispatch_group(v:vec3<u32>)->u32{{return v.x+v.y*65535u;}}\n{source}");
        let module = wgpu::naga::front::wgsl::parse_str(&source)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        println!("PASS arg-reduction shader validation; no GPU initialized");
        return Ok(());
    }
    let device = Device::new_vulkan(0)?;
    if !std::env::args().any(|a| a == "--bench-only") {
        for width in [7usize, 255, 256, 257, 4097, 248_320] {
            let mut rows = vec![vec![0f32; width]; 6];
            for (r, row) in rows.iter_mut().enumerate() {
                for (i, v) in row.iter_mut().enumerate() {
                    *v = ((i * 17 + r * 29) % 997) as f32 - 498.;
                }
            }
            rows[0][1] = f32::INFINITY;
            rows[0][width - 1] = f32::INFINITY;
            rows[1][2] = f32::NEG_INFINITY;
            rows[1][width - 1] = f32::NEG_INFINITY;
            rows[2][0] = f32::NAN;
            rows[2][1] = f32::INFINITY;
            rows[3][0] = 0.;
            rows[3][1] = f32::NAN;
            rows[4].fill(f32::NAN);
            rows[5].fill(-0.);
            let input = Tensor::from_vec(rows.concat(), (6, width), &device)?;
            for maximum in [true, false] {
                let got = if maximum {
                    input.argmax(1)?
                } else {
                    input.argmin(1)?
                }
                .to_vec1::<u32>()?;
                let expected: Vec<_> = rows.iter().map(|r| oracle(r, maximum)).collect();
                assert_eq!(got, expected, "width={width} maximum={maximum}");
                let offset = input.narrow(0, 1, 5)?;
                let got = if maximum {
                    offset.argmax(1)?
                } else {
                    offset.argmin(1)?
                }
                .to_vec1::<u32>()?;
                assert_eq!(got, expected[1..], "offset width={width} maximum={maximum}");
                // Strided fallback must have the same semantics.
                let strided = input.transpose(0, 1)?.contiguous()?.transpose(0, 1)?;
                let got = if maximum {
                    strided.argmax(1)?
                } else {
                    strided.argmin(1)?
                }
                .to_vec1::<u32>()?;
                assert_eq!(got, expected, "strided width={width} maximum={maximum}");
            }
            println!("PASS argmin/argmax width={width}: ties, infinity, NaN, signed zero, strided");
        }
    }
    let width = 248_320;
    let values: Vec<_> = (0..width).map(|i| ((i * 17) % 997) as f32).collect();
    let expected = oracle(&values, true);
    let x = Tensor::from_vec(values, (1, width), &device)?;
    for _ in 0..5 {
        let _ = x.argmax(1)?.to_vec1::<u32>()?;
    }
    let mut times = Vec::new();
    for _ in 0..25 {
        let start = Instant::now();
        let index = x.argmax(1)?.to_vec1::<u32>()?;
        assert_eq!(index, vec![expected]);
        times.push(start.elapsed().as_secs_f64() * 1e6);
    }
    times.sort_by(f64::total_cmp);
    println!(
        "argmax_248320_median_us={:.1} p95_us={:.1}",
        times[12], times[23]
    );
    Ok(())
}
