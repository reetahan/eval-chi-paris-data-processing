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
