@group(0) @binding(0) var<storage,read> input: array<f32>;
@group(0) @binding(1) var<storage,read_write> output: array<u32>;
var<workgroup> values: array<f32, __LANES__>;
var<workgroup> indices: array<u32, __LANES__>;

// Bit inspection deliberately avoids relying on x != x under fast math.
fn nan_bits(v:f32)->bool { return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u; }
fn better(a:f32,b:f32)->bool { return a __COMPARE__ b; }

@compute @workgroup_size(__LANES__)
fn main(@builtin(workgroup_id) wid:vec3<u32>, @builtin(local_invocation_index) lane:u32) {
    let row=dispatch_group(wid);
    if row>=__ROWS__u { return; }
    let base=__OFFSET__u+row*__WIDTH__u;
    let invalid=0xffffffffu;
    // Resolve a first NaN at the final store, so storage-dependent control flow
    // never bypasses a workgroup barrier.
    let first_nan=nan_bits(input[base]);
    var best=0.0;
    var index=invalid;
    for(var i=lane;i<__WIDTH__u;i+=__LANES__u) {
        let value=input[base+i];
        if !nan_bits(value) && (index==invalid || better(value,best)) {
            best=value;index=i;
        }
    }
    values[lane]=best;indices[lane]=index;
    workgroupBarrier();
    for(var stride=__HALF_LANES__u;stride>0u;stride/=2u) {
        if lane<stride {
            let other_index=indices[lane+stride];
            let other=values[lane+stride];
            let wins=indices[lane]==invalid || better(other,values[lane]) ||
                (other==values[lane] && other_index<indices[lane]);
            if other_index!=invalid && wins {
                values[lane]=other;indices[lane]=other_index;
            }
        }
        workgroupBarrier();
    }
    if lane==0u { output[row]=select(indices[0],0u,first_nan); }
}
