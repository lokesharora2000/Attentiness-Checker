# import packages
from keras.utils import np_utils
import numpy as np
import h5py

labels=[]
images=[]
class HDF5DatasetGenerator:

        # store the batch size, preprocessors, and data augmentor, whether or not
        # the labels should be binarized, along with the total number of classes
              
dbPath=config.TEST_HDF5
batchSize=config.BATCH_SIZE
aug = testAug
preprocessors = [iap]
classes = config.NUM_CLASSES
binarize = True
        
# open the HDF5 dataset for reading and determine the
# total number of entries in the database
db = h5py.File(dbPath)
numImages = db["labels"].shape[0]

passes = np.inf

# initialize the epoch count
epochs = 0

# keep looping infinitely -- the model will stop once we have
# reach the desired number of epochs
while epochs < passes:
    # loop over the HDF5 dataset
    for i in np.arange(0, numImages, batchSize):
        # extract the images and labels from the HDF5 dataset
        images = db["images"][i : i + batchSize]
        labels = db["labels"][i : i + batchSize]

        # check to see if the labels should be binarized
        if binarize:
            labels = np_utils.to_categorical(labels, classes)

        # check to see if our preprocessors are not None
        if preprocessors is not None:
            # initialize the list of processed images
            procImages = []

            # loop over the images
            for image in images:
                # loop over the preprocessors and apply each to the image
                for p in preprocessors:
                    image = p.preprocess(image)

                # update the list of preprocessed images
                procImages.append(image)

            # update the images array to be the processed images
            images = np.array(procImages)

        # if the data augmentator exists, apply it
        if aug is not None:
            (images, labels) = next(aug.flow(images, labels,
                batch_size = batchSize))

        # yield a tuple of images and labels
         

    # increment the total number of epochs
    epochs += 1

db.close()
