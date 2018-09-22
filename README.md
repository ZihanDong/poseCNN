# poseCNN

This repository is the implementation of the proposed "GoogLeNet Based Visual Camera Relative Pose Estimation".
We have provided the step-by-step procedure for model training, testing and evaluation.

## Download Dataset
Download the "Cleaned" data category from DTU-MVS dataset to your local path (138GB). (http://roboimagedata2.compute.dtu.dk/data/MVS/Cleaned.zip)

## Preprocess (optional but recommended)
This step will preprocess all the image (shrink and crop) and store them in another path with same file name and index.  Using the pre-processed images(224*224) could save a lot of memory space comparing to load/process original size images(1600*1200) while training.  As tested, around 4 GB memory is enough to load all the image samples (original size 138GB), which means we can avoid reading image from disk and thus make the training/testing much faster.

To perform preprocess, modify the directory and new\_dir in "down_sample_images.py" to the path for full size dataset and the location where you want to store the preprocessed images, also using the start and end index to specify the categories you want to be pre-processed, then run the script.

## For Training
Please open one of the "train_sub_sampled\_<settings>.py" in TrainTest folder and modify the following lines:

Line 65 to 67: Modify the prefix "D:/Dataset" to the path where you have your DTU dataset stored.

Line 20: Create an empty folder under "result" folder, keep the name identical to the "stteings" variable.

***For Training with full-size dataset, use "train.py" instead, however this could be very slow.

## For Testing
Open "test.py" (or "test_new_CNN.py" for different network archetecture) and add your training profile (string in "stteings" variable as mentioned above) to the setting list.  Then run the script with an input index as parameter to select your settings. (eg. "$python test.py 3" to select No.4 profile in setting list)

## For Evaluation
Open the notebook "Visualization&Evaluation.ipynb" and update the setting list as suggested above.  Then you can follow the instructions in comments to perform the translation/rotation error and error-by-category evaluation.

### Note that, we have provided 12 different profiles that already gone throgh testing, you can select the test result of these profiles for comparison while evaluating the model.
