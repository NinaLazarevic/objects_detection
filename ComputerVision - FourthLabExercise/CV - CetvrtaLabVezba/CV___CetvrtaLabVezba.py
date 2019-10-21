from helpers import pyramid
from helpers import sliding_window
import numpy as np
import argparse
import time
import cv2

# load the image and define the window width and height
image = cv2.imread("catanddog.jpg")
(winW, winH) = (128, 128)

rows = open("synset_words.txt").read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
length = len(classes)
net = cv2.dnn.readNetFromCaffe("bvlc_googlenet.prototxt", "bvlc_googlenet.caffemodel")

klasaPRED = [0] * length

for resized in pyramid(image, scale=1.5):
	# loop over the sliding window for each layer of the pyramid
	
	for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue

		#rows = open("synset_words.txt").read().strip().split("\n")
		#classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

		# our CNN requires fixed spatial dimensions for our input image(s)
		# so we need to ensure it is resized to 224x224 pixels while
		# performing mean subtraction (104, 117, 123) to normalize the input;
		# after executing this command our "blob" now has the shape:
		# (1, 3, 224, 224)
		blob = cv2.dnn.blobFromImage(window, 1, (224, 224), (104, 117, 123))

		# load our serialized model from disk
		print("[INFO] loading model...")
		#net = cv2.dnn.readNetFromCaffe("bvlc_googlenet.prototxt", "bvlc_googlenet.caffemodel")

		# set the blob as input to the network and perform a forward-pass to
		# obtain our output classification
		net.setInput(blob)
		start = time.time()
		preds = net.forward()
		end = time.time()
		print("[INFO] classification took {:.5} seconds".format(end - start))

		# sort the indexes of the probabilities in descending order (higher
		# probabilitiy first) and grab the top-5 predictions
		idxs = np.argsort(preds[0])[::-1][:5]

		# loop over the top-5 predictions and display them
		for (i, idx) in enumerate(idxs):
			# draw the top prediction on the input image
			if i == 0 and preds[0][idx] * 100 > 90 and klasaPRED[idx] == 0:
				text = "{}".format(classes[idx])
				cv2.putText(window, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
				klasaPRED[idx] = 1
				cv2.rectangle(resized, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
				
				
			# display the predicted label + associated probability to the
			# console	
			print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,	classes[idx], preds[0][idx]))
	
	
	
	cv2.imshow("Image", resized)
	cv2.waitKey(0)
	


		