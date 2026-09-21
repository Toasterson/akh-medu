@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read> weight:array<f32>;
@group(0) @binding(2) var<storage,read> history:array<f32>;
@group(0) @binding(3) var<storage,read_write> output:array<f32>;
@group(0) @binding(4) var<storage,read_write> next:array<f32>;

fn extended(batch:u32, position:u32, channel:u32)->f32 {
    if position<H {
        if HAS_HISTORY { return history[hi((batch*H+position)*C+channel)]; }
        return 0.0;
    }
    return input[xi((batch*L+position-H)*C+channel)];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=dispatch_index(id);
    if i<N {
        let channel=i%C;
        let token=(i/C)%L;
        let batch=i/(L*C);
        var sum=extended(batch,token,channel)*weight[wi(channel*K)];
        for(var tap=1u;tap<K;tap++) {
            sum+=extended(batch,token+tap,channel)*weight[wi(channel*K+tap)];
        }
        output[i]=sum;
    }
    // Histories are fresh allocations, not mutations of retained snapshots.
    // Width one only initializes the zero-length buffer's padded word;
    // it never evaluates a history-coordinate division by zero.
    __HISTORY_WRITE__
}
