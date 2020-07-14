# import packages
from config import emotion_config as config
from pipeline.preprocessing import ImageToArrayPreprocessor
from pipeline.io import HDF5DatasetGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import argparse
from sklearn.metrics import classification_report
import numpy as np 
!pip  install  scikit-plot

import scikitplot
# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type = str,
    help = "path to model checkpoint to load")
args = vars(ap.parse_args())

# initialize the testing dataset generator
    testAug = ImageDataGenerator(rescale = 1 / 255.0)
    iap = ImageToArrayPreprocessor()

# initialize the testing dataset generator
testGen = HDF5DatasetGenerator(config.TEST_HDF5, config.BATCH_SIZE,
    aug = testAug, preprocessors = [iap], classes = config.NUM_CLASSES )

# load the model from disk
print("[INFO] loading {}...".format(args["model"]))
model = load_model(args["model"])

# evaluate the network

from keras.models import load_model
model1 = load_model("epoch_75.hdf5")


(loss, acc) = model1.evaluate_generator(
    testGen.generator(),
    steps = testGen.numImages // config.BATCH_SIZE,
    max_queue_size = config.BATCH_SIZE * 2
)

# predIdxs = model1.predict(x=testGen, steps=( testGen.numImages// config.BATCH_SIZE))

# predIdxs = np.argmax(predIdxs, axis=1)

# Y_pred = model1.predict_generator(testGen ,steps = testGen.numImages // config.BATCH_SIZE)

predictions = model1.predict_generator(testGen.generator(),steps=testGen.numImages //config.BATCH_SIZE +1 ) 
#y_pred = np.argmax(Y_pred, axis=1) 
     
predictions = np.argmax(predictions, axis=1)


from sklearn.metrics import confusion_matrix
print('Confusion Matrix')                            

print(confusion_matrix(testLabels , predictions))

scikitplot.metrics.plot_confusion_matrix(testLabels , predictions)




print('Classification Report')

print(f'total wrong validation predictions: { np.sum(testLabels != predictions)}\n\n')

# show a nicely formatted classification report
print(classification_report(labels, predictions))


(loss, acc) = model1.evaluate_generator(
	testGen.generator(),
	steps=testGen.numImages // config.BATCH_SIZE,
	max_queue_size=config.BATCH_SIZE * 2)

print("[INFO] accuracy: {:.2f}".format(acc * 100))

# close the testing database
testGen.close()

print("[INFO] accuracy: {:.2f}".format(acc * 100))

# close the testing database
testGen.close()
