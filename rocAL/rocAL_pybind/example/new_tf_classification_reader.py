from amd.rali.plugin.tf import RALIIterator
from amd.rali.pipeline import Pipeline
import amd.rali.ops as ops
import amd.rali.types as types
import sys
import amd.rali.fn as fn
import tensorflow as tf


def main():
    if  len(sys.argv) < 5:
        print ('Please pass the <TensorFlowrecord> <cpu/gpu> <batch_size> <oneHotLabels=0/1> <display = True/False>')
        exit(0)
    imagePath = sys.argv[1]
    if(sys.argv[2] == "cpu"):
        raliCPU = True
    else:
        raliCPU = False
    bs = int(sys.argv[3])
    oneHotLabel = int(sys.argv[4])
    display = sys.argv[5]
    nt = 1
    di = 0
    cropSize = 224
    TFRecordReaderType = 0
    featureKeyMap = {
        'image/encoded':'image/encoded',
        'image/class/label':'image/class/label',
        'image/filename':'image/filename'
    }
    pipe = Pipeline(batch_size=bs, num_threads=nt,device_id=di, seed=2, rali_cpu=raliCPU)
    # pipe = HybridPipe(feature_key_map=featureKeyMap, tfrecordreader_type=TFRecordReaderType, batch_size=bs, num_threads=nt, device_id=di, data_dir=imagePath, crop=cropSize, oneHotLabels=oneHotLabel, rali_cpu=raliCPU)
    with pipe:
        inputs = fn.readers.tfrecord(path=imagePath, index_path = "", reader_type=TFRecordReaderType, user_feature_key_map=featureKeyMap,
            features={
                'image/encoded':tf.io.FixedLenFeature((), tf.string, ""),
                'image/class/label':tf.io.FixedLenFeature([1], tf.int64,  -1),
                'image/filename':tf.io.FixedLenFeature((), tf.string, "")
            }
        )
        jpegs = inputs["image/encoded"]
        images = fn.decoders.image(jpegs, user_feature_key_map=featureKeyMap, output_type=types.RGB, path=imagePath)
        resized = fn.resize(images, resize_x=300, resize_y=300)
        if(oneHotLabel == 1):
            labels = inputs["image/class/label"]
            labels = fn.one_hot(labels, num_classes=1000)
        pipe.set_outputs(resized)

    pipe.build()
    imageIterator = RALIIterator(pipe)
    for i in enumerate(imageIterator,0):
        print("\n\n",i)
    imageIterator.reset()



if __name__ == '__main__':
    main()
