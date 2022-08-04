export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib:/opt/rocm/rpp/lib
rm -rf ROCAL-CPU-RESULTS
mkdir ROCAL-CPU-RESULTS

../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests image_224x224 ROCAL-CPU-RESULTS/1-ROCAL-GPU-Rotate.png 224 224 2 0 1
../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests image_224x224 ROCAL-CPU-RESULTS/2-ROCAL-GPU-Brightness.png 224 224 3 0 1
../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests image_224x224 ROCAL-CPU-RESULTS/3-ROCAL-GPU-Flip.png 224 224 6 0 1
../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests image_224x224 ROCAL-CPU-RESULTS/4-ROCAL-GPU-Blur.png 224 224 7 0 1
../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests image_224x224 ROCAL-CPU-RESULTS/5-ROCAL-GPU-SaltAndPepperNoise.png 224 224 13 0 1
../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests image_224x224 ROCAL-CPU-RESULTS/6-ROCAL-GPU-Snow.png 224 224 14 0 1
../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests image_224x224 ROCAL-CPU-RESULTS/7-ROCAL-GPU-Pixelate.png 224 224 19 0 1
