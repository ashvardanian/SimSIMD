# SimSIMD

SIMD-accelerated similarity measures, metrics, distance functions for x86 and Arm.
Want to see how fast it runs?


```sh
cmake -DCMAKE_BUILD_TYPE=Release -B ./build_release && make -j2 -C ./build_release && ./build_release/simsimd_bench
```
