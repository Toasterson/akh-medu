// Four activation rows share each packed weight decode within a workgroup.
@group(0) @binding(0) var<storage,read> packed:array<u32>;
@group(0) @binding(1) var<storage,read> input:array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
var<workgroup> partial:array<vec4<f32>,64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let group=dispatch_group(wid);
    if group>=__GROUPS__u { return; }
    let row=group%__ROWS__u;
    let batch=(group/__ROWS__u)*4u;
    var sum=vec4<f32>(0.0);
    for(var col=lane;col<__COLS__u;col+=64u) {
        let w=weight(row*__COLS__u+col);
        var x=vec4<f32>(0.0);
        x.x=input[ii(batch*__COLS__u+col)];
        if batch+1u<__BATCH__u { x.y=input[ii((batch+1u)*__COLS__u+col)]; }
        if batch+2u<__BATCH__u { x.z=input[ii((batch+2u)*__COLS__u+col)]; }
        if batch+3u<__BATCH__u { x.w=input[ii((batch+3u)*__COLS__u+col)]; }
        sum+=w*x;
    }
    partial[lane]=sum;
    workgroupBarrier();
    for(var stride=32u;stride>0u;stride/=2u) {
        if lane<stride { partial[lane]+=partial[lane+stride]; }
        workgroupBarrier();
    }
    if lane==0u {
        output[batch*__ROWS__u+row]=partial[0].x;
        if batch+1u<__BATCH__u { output[(batch+1u)*__ROWS__u+row]=partial[0].y; }
        if batch+2u<__BATCH__u { output[(batch+2u)*__ROWS__u+row]=partial[0].z; }
        if batch+3u<__BATCH__u { output[(batch+3u)*__ROWS__u+row]=partial[0].w; }
    }
}
