@group(0) @binding(0) var<storage,read> input: array<f32>;
@group(0) @binding(1) var<storage,read> weight: array<f32>;
@group(0) @binding(2) var<storage,read_write> output: array<f32>;
var<workgroup> partial: array<f32,__LANES__>;

@compute @workgroup_size(__LANES__)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    let row = dispatch_group(group);
    if row >= __ROWS__u { return; }
    var sum = 0.0;
    for (var col = lane; col < __WIDTH__u; col += __LANES__u) {
        let x = input[xi(row * __WIDTH__u + col)];
        sum += x * x;
    }
    partial[lane] = sum;
    workgroupBarrier();
    for (var stride = __LANES__u / 2u; stride > 0u; stride /= 2u) {
        if lane < stride { partial[lane] += partial[lane + stride]; }
        workgroupBarrier();
    }
    let denominator = sqrt(partial[0] * bitcast<f32>(__INV_WIDTH__u)
        + bitcast<f32>(__EPS__u));
    for (var col = lane; col < __WIDTH__u; col += __LANES__u) {
        output[row * __WIDTH__u + col] =
            (input[xi(row * __WIDTH__u + col)] / denominator) * weight[wi(col)];
    }
}
