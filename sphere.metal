// File: sphere.metal
#include <metal_stdlib>
using namespace metal;

kernel void count_in_sphere(
        constant uint   &dim     [[ buffer(0) ]],
device   atomic_uint *cnt[[ buffer(1) ]],
constant uint   &seed0   [[ buffer(2) ]],
uint            tid     [[ thread_position_in_grid ]])
{
// Each thread handles one point at index = tid
uint id = tid;
// initialize RNG per thread
// mix tid and seed0 with linear congruential jumps
uint raw = tid * 747796405u + seed0 * 2891336453u + 1u;
uint seed = raw ^ (raw >> 16); // a simple xorshift mixer
const uint a = 1664525u, c = 1013904223u;
float sum = 0.0f;
for (uint d = 0; d < dim; ++d) {
seed = a * seed + c;
float x = (float)(seed & 0x7FFFFFFFu) / float(0x7FFFFFFF) * 2.0f - 1.0f;
sum += x * x;
}
if (sum <= 1.0f) {
atomic_fetch_add_explicit(cnt, 1u, memory_order_relaxed);
}
}
