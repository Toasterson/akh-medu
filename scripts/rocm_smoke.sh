#!/usr/bin/env bash
# ROCm/HIP GPU smoke test for akh-medu (ADR 053).
#
# Proves the AMD GPU actually executes compute via ROCm — not just that the libraries
# load: it runs a rocBLAS sgemm ON THE DEVICE and verifies the result against the analytic
# answer. This is the hardware foundation the Candle HIP backend (feature `rocm`) builds on.
#
# Prereqs (userspace only — no kernel driver, no reboot; see
# docs/ai/plans/2026-06-04-gpu-rocm-candle.md):
#   sudo apt-get install -y rocminfo rocm-hip-runtime rocblas hipblas hip-dev rocblas-dev
#
# Usage: scripts/rocm_smoke.sh
set -uo pipefail

ROCM="${ROCM_PATH:-/opt/rocm}"
[ -d "$ROCM" ] || { echo "ROCm not found at $ROCM (set ROCM_PATH). Install: see ADR 053."; exit 1; }

echo "== rocminfo: GPU agents =="
"$ROCM/bin/rocminfo" 2>/dev/null | grep -E 'Marketing Name:|gfx[0-9]' | sed 's/^/  /' | sort -u || {
  echo "  rocminfo found no GPU agent — check /dev/kfd and the render group."; exit 2; }

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
cat > "$tmp/gemm.cpp" <<'CPP'
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <vector>
#include <cmath>
#define CK(x) do{ auto e=(x); if(e){ printf("HIP err %d at %d\n",(int)e,__LINE__); return 2; } }while(0)
int main(){
  int n=512; size_t bytes=(size_t)n*n*sizeof(float);
  hipDeviceProp_t p; CK(hipGetDeviceProperties(&p,0));
  printf("device: %s  arch=%s  VRAM=%.1f GB\n", p.name, p.gcnArchName, p.totalGlobalMem/1e9);
  std::vector<float> A(n*n,1.0f), B(n*n,2.0f), C(n*n,0.0f);
  float *dA,*dB,*dC; CK(hipMalloc(&dA,bytes)); CK(hipMalloc(&dB,bytes)); CK(hipMalloc(&dC,bytes));
  CK(hipMemcpy(dA,A.data(),bytes,hipMemcpyHostToDevice));
  CK(hipMemcpy(dB,B.data(),bytes,hipMemcpyHostToDevice));
  rocblas_handle h; rocblas_create_handle(&h);
  float alpha=1.0f, beta=0.0f;
  rocblas_status s = rocblas_sgemm(h, rocblas_operation_none, rocblas_operation_none,
                                   n,n,n,&alpha,dA,n,dB,n,&beta,dC,n);
  if(s!=rocblas_status_success){ printf("rocblas sgemm status %d\n",(int)s); return 3; }
  CK(hipDeviceSynchronize());
  CK(hipMemcpy(C.data(),dC,bytes,hipMemcpyDeviceToHost));
  float expect=(float)n*1.0f*2.0f, got=C[0]; int ok=std::fabs(got-expect)<1e-3f;
  printf("sgemm %dx%d on GPU: C[0]=%.1f expected=%.1f -> %s\n", n,n, got, expect, ok?"PASS":"FAIL");
  rocblas_destroy_handle(h); (void)hipFree(dA); (void)hipFree(dB); (void)hipFree(dC);
  return ok?0:1;
}
CPP

echo "== compiling rocBLAS sgemm (system g++, HIP runtime API) =="
g++ -D__HIP_PLATFORM_AMD__ -I"$ROCM/include" "$tmp/gemm.cpp" \
    -L"$ROCM/lib" -lamdhip64 -lrocblas -o "$tmp/gemm" || { echo "compile failed"; exit 3; }

echo "== running on GPU =="
LD_LIBRARY_PATH="$ROCM/lib" "$tmp/gemm"
rc=$?
echo "== rocm_smoke exit: $rc =="
exit $rc
