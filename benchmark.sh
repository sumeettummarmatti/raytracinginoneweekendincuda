#!/bin/bash
set -e

echo "=== CUDA Ray Tracing Benchmark ==="
echo "Detecting GPUs..."
nvidia-smi -L 2>/dev/null || echo "(nvidia-smi not found)"
echo ""

# ── Build baseline ────────────────────────────────────────────────────────────
echo "=> Building Baseline (Makefile)..."
make clean > /dev/null 2>&1 || true
make cudart 2>&1 | tail -3
echo "Baseline built."

# ── Build accelerated ─────────────────────────────────────────────────────────
echo "=> Building Accelerated (CMake)..."
rm -rf build && mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | grep -E "OIDN|arch|Error" || true
make -j$(nproc) 2>&1 | tail -5
cd ..
echo "Accelerated built."
echo ""

# ── Run baseline ──────────────────────────────────────────────────────────────
echo "=> Running Baseline (1200x800, 10 spp)..."
{ time ./cudart > baseline.ppm ; } 2>&1 | tee baseline_time.txt || true
echo ""

# ── Run accelerated ───────────────────────────────────────────────────────────
echo "=> Running Accelerated (1200x800, 10 spp)..."
./build/rt_accel --width 1200 --height 800 --spp 10 > accel.ppm 2> accel_stderr.log || true
echo ""

# ── Summarise ─────────────────────────────────────────────────────────────────
echo "========================================="
echo " RESULTS"
echo "========================================="
echo "Baseline timing:"
grep -i "took\|real" baseline_time.txt 2>/dev/null || echo "  (not found)"

echo ""
echo "Accelerated timing:"
if [ -f accel_time.txt ]; then
    cat accel_time.txt
else
    grep "TIMING" accel_stderr.log 2>/dev/null || echo "  (accel_time.txt not written — check accel_stderr.log)"
fi

echo ""
echo "LBVH diagnostics (from accelerated run):"
grep -E "LBVH|ROOT|sphere\[0\]|bounce1|GPU" accel_stderr.log 2>/dev/null | head -20 || true

echo ""
echo "Output images: baseline.ppm  accel.ppm"
echo ""
echo "Quick pixel sanity check (non-sky pixels have R < 200 in many places):"
if command -v python3 &>/dev/null; then
python3 - <<'PYEOF'
import sys, struct

def check_ppm(path, label):
    try:
        with open(path, 'r') as f:
            assert f.readline().strip() == 'P3'
            W, H = map(int, f.readline().split())
            maxv  = int(f.readline())
            vals  = list(map(int, f.read().split()))
        sky_count = 0
        geo_count = 0
        for idx in range(0, len(vals), 3):
            r, g, b = vals[idx], vals[idx+1], vals[idx+2]
            if r < 220 or g < 200:   # rough geometry detector
                geo_count += 1
            else:
                sky_count += 1
        total = (W * H)
        print(f"  {label}: {W}x{H}  geometry≈{geo_count/total*100:.1f}%  sky≈{sky_count/total*100:.1f}%")
    except Exception as e:
        print(f"  {label}: could not read — {e}")

check_ppm("baseline.ppm", "Baseline")
check_ppm("accel.ppm",    "Accelerated")
PYEOF
fi

echo ""
echo "To profile warp efficiency:"
echo "  ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio \\"
echo "      ./build/rt_accel --width 400 --height 300 --spp 2 > /dev/null"
echo ""
echo "To check FP64 leaks (should be 0):"
echo "  ncu --metrics smsp__sass_thread_inst_executed_op_dfma_pred_on.sum \\"
echo "      ./build/rt_accel --width 400 --height 300 --spp 2 > /dev/null"
