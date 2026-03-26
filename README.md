# web_perceptron_hw
This file provides a very brief explanation of what everything in the repository does. A more comprehensive report was submitted to Brightspace.

## index.html
This is the actual webpage. It uses the weights.npy and bias.npy files to operate. Technically, the page only needs these three files to work, but everything else used is included so its all in one place.

## createnpy.py
This converts the image files from ./LTdata into usable labels and features that are saved to X.npy and y.npy. I do not know if this file works opn github since I ran it locally on my machine.

## perceptron.py
This was also ran locally. It trains off of the X and y files created earlier and outputs weights.npy and bias.npy which is used by the webpage image prediction.

## LTdata
This directory includes the 100 samples that I created. The are put into seperate directories based on whether the image is an L or a T.
