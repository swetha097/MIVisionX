#!/bin/bash

############# Edit GDF path and file names #############
GDF_PATH="../../../vision_tests/gdfs"
GDF_FILE_LIST="01_absDiff.gdf 02_accumulate.gdf 03_accumulateSquared.gdf 04_accumulateWeighted.gdf 05_add.gdf 06_and.gdf 09_channelCombine.gdf 23_magnitude.gdf 
                27_multiply.gdf 28_not.gdf 30_or.gdf 31_phase.gdf 36_subtract.gdf 37_tableLookup.gdf 38_threshold.gdf 41_xor.gdf"
AFFINITY_LIST="CPU GPU"
############# Edit GDF path and file names #############

for AFFINITY in $AFFINITY_LIST;
do
    printf "\n\n---------------------------------------------"
    printf "\nRunning GDF cases on runvx for $AFFINITY..."
    printf "\n---------------------------------------------\n"
    for GDF_FILE in $GDF_FILE_LIST;
    do
        
        printf "\n$GDF_FILE..."
        runvx -affinity:$AFFINITY $GDF_PATH/$GDF_FILE
    done
done