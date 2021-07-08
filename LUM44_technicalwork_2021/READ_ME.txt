Read Me File

General Description:
Code and accompanying datasets used to detect anoamlies and impute new data for fish movement data.
By Luke Mullis

#############Please note the full_library2 and full_library are the main programs in this project #####################

Software required:
Python compiler


Pythion libraries required:
numpy
pandas
matplotlib.pyplot
matplotlib.dates
statistics
math
random
sklearn
cv2


Datasets:
####genreal#####
the 'perfect_fish' contains perfect human tacked data
the 'anom_data' is the data from 'perfect_fish' sets but with anomalies added
the 'anom' data sets contain the location and the size of the anoamlies in the 'anom_data' sets
################


First test data set:
anom_30_1.csv
anom_data_30_1.csv
perfect_fish_30_0

Second data set:
anom_30_2.csv
anom_data_30_2.csv
perfect_fish_30_1

First data set coverted to 7 points
anom_7_1.csv
anom_data_7_1.csv
perfect_fish_7_0

Second data set coverted to 7 points:
anom_7_2.csv
anom_data_7_2.csv
perfect_fish_7_1


6 perfect human tracked data sets each trackig 7 points
perfect_fish_7_0
perfect_fish_7_1
perfect_fish_7_2
perfect_fish_7_3
perfect_fish_7_4
perfect_fish_7_5

6 perfect human tracked data sets each trackig 30 points
perfect_fish_30_0
perfect_fish_30_1
perfect_fish_30_2
perfect_fish_30_3
perfect_fish_30_4
perfect_fish_30_5



#####real world data######
the 'deeplabcut' data contains data collected from the deeplabcut algorithm
the 'real_world_7' data is human tracked data thracking the same fish as the 'deeplabcut' sets
the 'normalised' data sets are the 'deeplabcut' 'real_world_7' datasets normalised to th elength of the fish
the 'anoamly_detected' is the 'normalised' data with the anomamlies replaced with 'nan'
##########################

first data set
real_world_1_deeplabcut
real_world_7_1
real_world_7_deeplab_normalised_1
real_world_7_normalised_1
real_world_7_normalised_anomaly_detected_1

second data set
real_world_2_deeplabcut
real_world_7_2
real_world_7_deeplab_normalised_2
real_world_7_normalised_2
real_world_7_normalised_anomaly_detected_2





programs:
full_library: All the collective anomaly functions. This file is imported into other files 
full_library2: All the contexual anomaly functions. This file is imported into other files

collective_anomaly_detection_test : tests the success of collective anomaly detection code
collective_anomaly_test: tests the collective anoamly detection and imputation algorithms
collective_imputation_test: testing collective imputation technique on the 30 point data sets
contexual_anomaly_detection_code : tests success of contexual anomaly detection code

imputation_30: performs imputation on the sets tracking 30 points (dataset1)
imputation_30_2: performs imputation on the sets tracking 30 points (dataset2)
y_imputation_test: performs imputation by imputing x points and y points seperatly

covert_30_to_7: convertes the sets tracking 30 points to sets tracking 7
imputation_7: performs inputation on the sets tracking 7 points (dataset2)
imputation_7_2: performs inputation on the sets tracking 7 points (dataset1)

prepare_dataset: used to add anoamlies to a perfect data set

normalise_real_world_1: used to normalise the datasets
real_world_anomaly_detection: used to detect the anomalies in the real world data sets
real_world_data_test: imputation test on the anomaly detected real world data set 1
real_world_data_test_2 : imputation test on the anomaly detected real world data set 2

video_test_1 : The fish video with points tracked upon it.




















































