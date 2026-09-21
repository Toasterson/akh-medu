// Two 32-lane partitions per row, eight weights per lane per GGUF block.
// Preserve the packed artifact bytes; reuse scales and quant bytes across values.
@group(0) @binding(0) var<storage,read> packed:array<u32>;
@group(0) @binding(1) var<storage,read> input:array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
var<workgroup> partial:array<f32,64>;

fn scale_min(base:u32,g:u32)->vec2<f32> {
    if g<4u {
        return vec2<f32>(f32(byte_at(base+4u+g)&63u),f32(byte_at(base+8u+g)&63u));
    }
    let s=(byte_at(base+8u+g)&15u)|((byte_at(base+g)>>6u)<<4u);
    let m=(byte_at(base+8u+g)>>4u)|((byte_at(base+4u+g)>>6u)<<4u);
    return vec2<f32>(f32(s),f32(m));
}

fn activation4(base:u32,stride:u32)->vec4<f32> {
    return vec4<f32>(input[ii(base)],input[ii(base+stride)],
                     input[ii(base+2u*stride)],input[ii(base+3u*stride)]);
}

fn dot_q4(block:u32,col:u32,lane:u32)->f32 {
    let base=block*144u;
    let dm=unpack2x16float(packed[base/4u]);
    let bytes=vec4<u32>(byte_at(base+16u+lane),byte_at(base+48u+lane),
                        byte_at(base+80u+lane),byte_at(base+112u+lane));
    let s0=scale_min(base,0u);let s1=scale_min(base,1u);
    let s2=scale_min(base,2u);let s3=scale_min(base,3u);
    let s4=scale_min(base,4u);let s5=scale_min(base,5u);
    let s6=scale_min(base,6u);let s7=scale_min(base,7u);
    let low=(dm.x*vec4<f32>(s0.x,s2.x,s4.x,s6.x))*vec4<f32>(bytes&vec4<u32>(15u))
             -dm.y*vec4<f32>(s0.y,s2.y,s4.y,s6.y);
    let high=(dm.x*vec4<f32>(s1.x,s3.x,s5.x,s7.x))*vec4<f32>(bytes>>vec4<u32>(4u))
              -dm.y*vec4<f32>(s1.y,s3.y,s5.y,s7.y);
    return dot(low,activation4(col+lane,64u))+dot(high,activation4(col+lane+32u,64u));
}

fn signed_scale(offset:u32)->f32 {
    return f32(i32(byte_at(offset)^128u)-128);
}

fn dot_q6_half(base:u32,col:u32,lane:u32,half:u32,d:f32)->f32 {
    let lo0=byte_at(base+half*64u+lane);
    let lo1=byte_at(base+half*64u+32u+lane);
    let hi=byte_at(base+128u+half*32u+lane);
    let low=vec4<u32>(lo0&15u,lo1&15u,lo0>>4u,lo1>>4u);
    let high=vec4<u32>(hi,hi>>2u,hi>>4u,hi>>6u)&vec4<u32>(3u);
    let quant=vec4<f32>(vec4<i32>(low|(high<<vec4<u32>(4u)))-vec4<i32>(32));
    let s=base+192u+half*8u+lane/16u;
    let scales=vec4<f32>(signed_scale(s),signed_scale(s+2u),signed_scale(s+4u),signed_scale(s+6u));
    return dot((d*scales)*quant,activation4(col+half*128u+lane,32u));
}

fn dot_q6(block:u32,col:u32,lane:u32)->f32 {
    let base=block*210u;
    let d=unpack2x16float(byte_at(base+208u)|(byte_at(base+209u)<<8u)).x;
    return dot_q6_half(base,col,lane,0u,d)+dot_q6_half(base,col,lane,1u,d);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let row=dispatch_group(wid);
    if row>=__ROWS__u { return; }
    var sum=0.0;
    for(var block=lane/32u;block<__BLOCKS__u;block+=2u) {
        sum+=__DOT__(row*__BLOCKS__u+block,block*256u,lane%32u);
    }
    partial[lane]=sum;
    workgroupBarrier();
    for(var stride=32u;stride>0u;stride/=2u) {
        if lane<stride { partial[lane]+=partial[lane+stride]; }
        workgroupBarrier();
    }
    if lane==0u { output[row]=partial[0]; }
}
