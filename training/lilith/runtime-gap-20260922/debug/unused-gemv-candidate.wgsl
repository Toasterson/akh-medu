// One row, 32 output columns and eight independent reduction lanes per group.
// ai/bi retain offsets, broadcasts and transpose strides supplied by Layout.
@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read> b:array<f32>;
@group(0) @binding(2) var<storage,read_write> out:array<f32>;
var<workgroup> partial:array<f32,256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_index) lane:u32) {
    let tile=dispatch_group(group);
    if tile >= BATCH*TILES { return; }
    let batch=tile/TILES;
    let col=(tile%TILES)*32u+lane%32u;
    let reduction_lane=lane/32u;
    let a_base=ai(batch*K);
    let b_base=bi(batch*K*N)+col*B_COL_STRIDE;
    var sum=0.0;
    if col<N {
        for(var j=reduction_lane;j<K;j+=8u) {
            sum+=a[a_base+j*A_COL_STRIDE]*b[b_base+j*B_ROW_STRIDE];
        }
    }
    partial[lane]=sum;
    workgroupBarrier();
    if reduction_lane==0u && col<N {
        var value=0.0;
        for(var r=0u;r<8u;r++) { value+=partial[r*32u+lane]; }
        out[batch*N+col]=value;
    }
}
