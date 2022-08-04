export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib:/opt/rocm/rpp/lib
rm -rf ROCAL-GPU-RESULTS
mkdir ROCAL-GPU-RESULTS

../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests image_224x224 ROCAL-GPU-RESULTS/1-ROCAL-GPU-Rotate.png 224 224 2 1 1
../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests image_224x224 ROCAL-GPU-RESULTS/2-ROCAL-GPU-Brightness.png 224 224 3 1 1
../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests image_224x224 ROCAL-GPU-RESULTS/3-ROCAL-GPU-Flip.png 224 224 6 1 1
../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests image_224x224 ROCAL-GPU-RESULTS/4-ROCAL-GPU-Blur.png 224 224 7 1 1
../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests image_224x224 ROCAL-GPU-RESULTS/5-ROCAL-GPU-SnPNoise.png 224 224 13 1 1
../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests image_224x224 ROCAL-GPU-RESULTS/6-ROCAL-GPU-Snow.png 224 224 14 1 1
../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests image_224x224 ROCAL-GPU-RESULTS/7-ROCAL-GPU-Pixelate.png 224 224 19 1 1
