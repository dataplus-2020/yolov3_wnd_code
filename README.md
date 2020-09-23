# Ultralytics YOLOv3
For general information about using the Ultralytics YOLOv3, please visit https://github.com/ultralytics/yolov3.

# Description
For our experiment, we trained this model on overhead imagery to detect wind turbines. We also created synthetic overhead imagery to supplement our training data set. Because of this, we have added many different preprocessing scripts that are meant for the real and synthetic overhead imagery that are not found with the ultralytics yolov3.

# Uploading Data
Create a folder for the data, such as yolov3_wnd_code/data. For real imagery, this should be set of images and formatted labels in a folder such as yolov3_wnd_code/data/images_and_labels. The labels should be formatted with each line as: class x-coordinate y-coordinate height width. These values are proportions relative to the size of the image rather than pixel values and the x-coordinate and y-coordinate are for the center of the bounding box. Each line corresponds to a single bounding box. You also need to upload a .NAMES file that has lists the names of the classes you are trying to detect (an example of this can be found in data_wnd)

<img src="https://github.com/dataplus-2020/yolov3_wnd_code/blob/master/example_label.png" height="50%" width="50%">

# Preprocessing

## Real Overhead Imagery
Our real overhead imagery was 1114x1114 and 1114x1115 images that we hand-labeled. To improve efficiency and to get images of the same exact dimensions, we split each image into four 608x608 images. In utils/data_process_distribution_vis_util/preprocess_wv_for_windturbine.py, this script takes these large overhead images and their labels and extracts the four 608x608 images with their corresponding labels. It also splits the data randomly into training and validation and creates a .DATA file that refers to the training and validation files. To use this script with your custom data, simply change the paths in the get_args() function of the file to point to the data folder you have uploaded. For more info, read the documentation for the functions in preprocess_wv_for_windturbinie.py.

## Synthetic Overhead Imagery
After creating synthetic imagery using CityEngine, we had a set of 608x608 RGB images with corresponding black and white segmented images that provided the locations of the wind turbines. In these black and white images, the wind turbines were colored black. The preprocessing script, utils/synthetic_data_preprocess/preprocess_syn_xview_background_gt_labels.py is used to convert these black and white images into formatted labels. A function in this script also splits the data into training and validation and creates a .DATA file that refers to those files, however, this would include synthetic data in the validation set. As is, the function will create two .txt files that list the paths for the synthetic images and labels. Afterwards, you can simply copy and paste these into training set image and label .txt files that you want to supplement. To use this script with your custom data, simply change the arguments of the file to point to the data folder you have uploaded. For more info, read the documentation in preprocess_syn_xview_background_gt_labels.py.

For examples of the four .txt files and the .data file, you can look in data_wnd/real

# Training
Before training, you should have:
* Uploaded your data and preprocessed to you have a set of images of the correct size and corresponding labels
* .data file that lists the paths of the four .txt files
* Four .txt files that list paths: training images, training labels, validation images, validation labels
* A .NAMES file that lists the names of the classes
* .json file that provides the path of the .data file in train_cfg (you can also change the batch size and num of epochs here)  

The final step is to edit train.sh, and make sure that the cfg_dict path is set to the .json file you want to use. You can additionally change the hyperparameters in train_syn_xview_background_1cls_mean_best_example.py before training.
Then, you can run `bash train.sh` and it will start training!

After the testing finishes, the results will be in result_output. There will be a PR curve as well as a curve for the different loss values.

# Testing
Use test_xview.py. Before running, change the weights and data arguments to point to the correct files.

# Visualizing the Output
Use detect_xview.py. Before running, change the weights and data arguments to point to the correct files.


