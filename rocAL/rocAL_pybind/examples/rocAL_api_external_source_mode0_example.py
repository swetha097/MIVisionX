from random import shuffle
from amd.rocal.pipeline import Pipeline
from amd.rocal.plugin.pytorch import ROCALClassificationIterator
import amd.rocal.fn as fn
import amd.rocal.types as types
import os

def main():
    batch_size = 3
    data_dir = os.environ["ROCAL_DATA_PATH"] + "/coco/coco_10_img/train_10images_2017/"
    device = "cpu"
    try:
        path = "OUTPUT_IMAGES_PYTHON/EXTERNAL_SOURCE_READER/MODE0/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)

    def image_dump(image, idx, device="cpu"):
        import cv2
        if device == "gpu":
            image = image.cpu().detach().numpy()
        else:
            image = image.detach().numpy()
        image = image.transpose([1, 2, 0]) # NCHW
        image = (image).astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite("OUTPUT_IMAGES_PYTHON/EXTERNAL_SOURCE_READER/MODE0/" + str(idx)+"_"+"train"+".png", image)

    #Define the Data Source for all image samples - User needs to define their own source
    class ExternalInputIteratorMode0(object):
        def __init__(self, batch_size):
            self.images_dir = data_dir
            self.batch_size = batch_size
            self.files = []
            import glob
            for filename in glob.glob(os.path.join(self.images_dir, '*.jpg')):
                self.files.append(filename)
            shuffle(self.files)

        def __iter__(self):
            self.i = 0
            self.n = len(self.files)
            return self

        def __next__(self):
            batch = []
            label = []
            for _ in range(self.batch_size):
                jpeg_filename = self.files[self.i]
                batch.append(jpeg_filename)
                label.append(1) # Label is some random variable for testing - user can modify acording to use case
                self.i = (self.i + 1) % self.n
            return batch, label

    # Mode 0
    external_input_source = ExternalInputIteratorMode0(batch_size)

    #Create the pipeline
    external_source_pipeline_mode0 = Pipeline(batch_size=batch_size, num_threads=1, device_id=0, seed=1, rocal_cpu=True if device=="cpu" else False, tensor_layout=types.NCHW)

    with external_source_pipeline_mode0:
        jpegs, _ = fn.external_source(source=external_input_source, mode=types.EXTSOURCE_FNAME)
        output = fn.resize(jpegs, resize_width=300, resize_height=300, output_layout=types.NCHW, output_dtype=types.UINT8)
        external_source_pipeline_mode0.set_outputs(output)

    # build the external_source_pipeline_mode0
    external_source_pipeline_mode0.build()
    #Index starting from 0
    cnt = 0
    # Dataloader
    data_loader = ROCALClassificationIterator(external_source_pipeline_mode0, device=device)
    for i, output_list in enumerate(data_loader, 0):
            print("**************", i, "*******************")
            print("**************starts*******************")
            print("\nImages:\n", output_list)
            print("**************ends*******************")
            print("**************", i, "*******************")
            for img in output_list[0]:
                cnt = cnt + 1
                image_dump(img, cnt, device=device)

if __name__ == '__main__':
    main()
