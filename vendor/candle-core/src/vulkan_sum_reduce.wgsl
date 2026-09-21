// Cooperative f32 sum for contiguous rows; accumulation stays f32.
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read_write> output:array<f32>;
var<workgroup> partial:array<f32,__LANES__>;

@compute @workgroup_size(__LANES__)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let row=dispatch_group(wid);
    if row>=__ROWS__u { return; }
    let base=__OFFSET__u+row*__WIDTH__u;
    var sum=0.0;
    for(var col=lane;col<__WIDTH__u;col+=__LANES__u) { sum+=input[base+col]; }
    partial[lane]=sum;
    workgroupBarrier();
    for(var stride=__HALF_LANES__u;stride>0u;stride/=2u) {
        if lane<stride { partial[lane]+=partial[lane+stride]; }
        workgroupBarrier();
    }
    if lane==0u { output[row]=partial[0]; }
}
