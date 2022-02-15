# eval-chi-paris-data-processing

This repository contains the data and necessary scripts needed to generate the observational data and run the model used in the paper, Machine Learning Model Evaluation for Estimating Submicron Aerosol Mixing State at the Global Scale.
Make sure you have Python 3 installed. You can then ensure that you have all necessary packages installed by running
```
pip install -r requirements.txt
```
In any of the following commands, you can choose to not provide an output filename, the program will simply use a .
# **How to Generate the Observational Dataset**
Simply run
```
python observ_processing.py [-o your_output_name.csv]
```
to have the raw observational data processing script generate the final observed data CSV, with a default output filename. You can use the -o option and add an output filename of your choice. The raw data files are stored in the data folder.

# **How To Run the Model and Evaluate Results**
Assuming you have generated training data in a singular CSV file, you can run training, validation and testing by running
```
python model.py path_to_training_data [-h] [-ov your_validation_output_name.csv] [-ot your_testing_output_name.csv]
```

If you wish to run hyperparameter tuning yourself instead of using the values we found, you can do so with the -h option. You can use the -ov option and -ot option, to add a validation output filename and testing output filename of your choice, respectively, if you do not want the default. You MUST provide the path to your training data file (make sure it is contained in a singular CSV!)

To evaluate the results, simply run 
```
python evaluate.py path_to_validation_output.csv path_to_testing_output.csv [-om your_metrics_output_name.txt] [-ov your_validation_fig_name.csv] [-ot your_testing_fig_name.csv]
```
which will display the computed metric in your terminal and pop up the figures. You can also save all of these results to their own files by using the -om,-ov and -ot options to save the metrics, validation results and testing results, respectively. You MUST provide the path to your validation and testing outputs from model.py, with the validation first.
