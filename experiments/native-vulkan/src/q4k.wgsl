// Exact Candle/GGUF Q4_K layout: 144 bytes per 256 weights.
@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> shape: vec4<u32>;
var<workgroup> partial: array<f32, 64>;

fn byte_at(offset: u32) -> u32 {
    return (packed[offset / 4u] >> ((offset % 4u) * 8u)) & 255u;
}

fn weight_q4k(block: u32, element: u32) -> f32 {
    let base = block * 144u;
    let dm = unpack2x16float(packed[base / 4u]);
    let group = element / 32u;
    var scale: u32;
    var minimum: u32;
    if group < 4u {
        scale = byte_at(base + 4u + group) & 63u;
        minimum = byte_at(base + 8u + group) & 63u;
    } else {
        scale = (byte_at(base + 8u + group) & 15u) | ((byte_at(base + group) >> 6u) << 4u);
        minimum = (byte_at(base + 8u + group) >> 4u) | ((byte_at(base + 4u + group) >> 6u) << 4u);
    }
    let qbyte = byte_at(base + 16u + (element / 64u) * 32u + element % 32u);
    let quant = (qbyte >> ((group % 2u) * 4u)) & 15u;
    return (dm.x * f32(scale)) * f32(quant) - dm.y * f32(minimum);
}

// Exact Q6_K: ql[128], qh[64], signed scales[16], f16 d = 210 bytes.
// Blocks alternate between 32-bit aligned and half-word aligned starts.
fn weight_q6k(block: u32, element: u32) -> f32 {
    let base = block * 210u;
    let d_bits = byte_at(base + 208u) | (byte_at(base + 209u) << 8u);
    let d = unpack2x16float(d_bits).x;
    let half = element / 128u;
    let part = (element % 128u) / 32u;
    let lane = element % 32u;
    let low = byte_at(base + half * 64u + (part % 2u) * 32u + lane);
    let high = byte_at(base + 128u + half * 32u + lane);
    let quant = i32(((low >> ((part / 2u) * 4u)) & 15u) | (((high >> (part * 2u)) & 3u) << 4u)) - 32;
    let scale_byte = byte_at(base + 192u + half * 8u + part * 2u + lane / 16u);
    let scale = i32(scale_byte ^ 128u) - 128;
    return (d * f32(scale)) * f32(quant);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let row = wid.x;
    var sum = 0.0;
    for (var col = lane; col < shape.y; col += 64u) {
        let block = row * (shape.y / 256u) + col / 256u;
        var w: f32;
        if shape.z == 6u { w = weight_q6k(block, col % 256u); }
        else { w = weight_q4k(block, col % 256u); }
        sum += w * input[col];
    }
    partial[lane] = sum;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if lane < stride { partial[lane] += partial[lane + stride]; }
        workgroupBarrier();
    }
    if lane == 0u { output[row] = partial[0]; }
}
