# eval-chi-paris-data-processing

This repository contains the data and necessary scripts needed to generate the observational data and run the model used in the paper, [].
Make sure you have Python 3 installed. You can then ensure that you have all necessary packages installed by running
```
pip install -r requirements.txt
```
# **How to Generate the Observational Dataset**
Simply run
```
python observ_processing.py your_output_name.csv
```
to have the raw observational data processing script generate the final observed data CSV, with a filename of your choice. The raw data files are stored in the data folder.

# **How To Run the Model and Evaluate Results**
Assuming you have generated training data in a singular CSV file, you can run training, validation and testing by running
```
python model.py path_to_training_data your_validation_output_name.csv your_testing_output_name.csv
```
If you wish to run hyperparameter tuning yourself instead of using the values we found, you can do so with the -h option, so the command would be
```
python model.py path_to_training_data your_validation_output_name.csv your_testing_output_name.csv
```
To evaluate the results, simply run 
```
python evaluate.py path_to_validation_output.csv path_to_testing_output.csv
```
which will display the computed metric in your terminal and pop up the figures. You can also save all of these results to their own files by running
```
python evaluate.py path_to_validation_output.csv path_to_testing_output.csv your_metrics_output_name.txt your_validation_fig_name.csv your_testing_fig_name.csv
```
