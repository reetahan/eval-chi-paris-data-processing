import xgboost as xgb
from xgboost import XGBRegressor
from sklearn import model_selection, metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import pickle
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import gc
import matplotlib.colors as clr
import warnings
import sys
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


def hyper_tune(x_train, y_train, n_iter):
    sys.stdout.flush()
    print('Begin hypertuning')

    s_lr = 0.01
    e_lr = 0.04
    s_n = 1050
    e_n = 1400
    s_md = 7
    e_md = 10
    hit_bound = True

    while(hit_bound):
        print('New boundaries: LR - [' + str(s_lr) + ',' + str(e_lr) + '] || N_EST - [' + str(
            s_n) + ',' + str(e_n) + '] || MD - [' + str(s_md) + ',' + str(e_md) + ']')
        opt = BayesSearchCV(
            XGBRegressor(objective='reg:squarederror', n_jobs=1),
            {
                'learning_rate': Real(s_lr, e_lr),
                'n_estimators': Integer(s_n, e_n),
                'max_depth': Integer(s_md, e_md)
            },
            scoring='neg_mean_squared_error',
            n_iter=n_iter,
            verbose=2,
            cv=6,
            n_points=9,
            n_jobs=54,
            random_state=66
        )
        _ = opt.fit(x_train, y_train)

        print("Current best hyperparameters!")
        for k in opt.best_params_:
            print(f"{k}: {opt.best_params_[k]}")

        final_lr = opt.best_params_['learning_rate']
        final_n = opt.best_params_['n_estimators']
        final_md = opt.best_params_['max_depth']

        hit_bound = False
        print("Bound check")
        if(final_lr == s_lr or final_lr == e_lr):
            hit_bound = True
            print("Boundary hit - LR")
            if(final_lr == s_lr):
                e_lr = s_lr
                if(s_lr > 0.05):
                    s_lr = s_lr - 0.05
                else:
                    s_lr = s_lr - 0.006
            else:
                s_lr = e_lr
                e_lr = e_lr + 0.05

        if(final_n == s_n or final_n == e_n):
            hit_bound = True
            print('Boundary hit - N_EST')
            if(final_n == s_n):
                e_n = s_n
                s_n = s_n - 200
            else:
                s_n = e_n
                e_n = e_n + 200

        if(final_md == s_md or final_md == e_md):
            hit_bound = True
            print('Boundary hit - MD')
            if(final_md == s_md):
                e_md = s_md
                s_md = s_md - 2
            else:
                s_md = e_md
                e_md = e_md + 2

        if(not hit_bound):
            print('Boundary avoidance successful')
            return final_lr, final_n, final_md



def feat_imp(XGBreg):
    results = XGBreg.get_booster().get_score()
    total = 0
    for res in results:
        total = total + results[res]
    for res in results:
        results[res] = results[res]/total * 100
    print("Feature Importance Shares")
    for res in results:
        print(res + ": " + str(results[res]) + "%")

def usage():
    usage_str = 'Usage: python model.py path_to_training_data path_to_validation_data path_to_testing_data [-h] [-ov your_validation_output_name.csv] [-ot your_testing_output_name.csv]'
    print(usage_str)
    sys.exit()

def ml_workflow_grid_search(df_train, df_validate, df_test, features_ls, pred, time_label, is_tune, val_outfile, test_outfile):

    param_grid = para_dict
    XGBreg = XGBRegressor(objective='reg:squarederror', n_jobs=36)

    X_train = df_train[features_ls]
    y_train = df_train[pred]
    X_validate = df_validate[features_ls]
    y_validate = df_validate[pred]
    X_test = df_test[features_ls]
    y_test = df_test[pred]

    print('Data separated')

    '''
    kfold = model_selection.KFold(n_splits=10, random_state=7)

    '''
    learning_rate_bayes = 0.0123748445508538
    n_estimators_bayes = 1088
    max_depth_bayes = 8
    if(is_tune):
        learning_rate_bayes, n_estimators_bayes, max_depth_bayes = hyper_tune(X_train, y_train, n_iter=128)

    # Get the models
    XGBreg = XGBRegressor(objective='reg:squarederror',
                          learning_rate=learning_rate_bayes,
                          max_depth=max_depth_bayes,
                          n_estimators=n_estimators_bayes, n_jobs=36, verbosity=1)
    print('Model setup')
    XGBreg.fit(X_train, y_train)
    print('Model trained')

 # Run model on training set

    df_predictions = XGBreg.predict(X_train)
    print('Tested on training set')
    
    f_train = open("3_train_res_800_3.txt", "w+")
    f_train.write("Training predictions\n")
    for entr in df_predictions:
        f_train.write(str(entr) + "\n")
    f_train.write("Y_train\n")
    for entr in y_train:
        f_train.write(str(entr) + "\n")
    f_train.close()
    

    # Run model on validation set
    df_predictions = XGBreg.predict(X_validate)
    print('Ran model on validation set')

    #val_outfile = "3_validation_res_800_3.txt"
    f_val = open(val_outfile, "w+")
    f_val.write("Validation predictions\n")
    for entr in df_predictions:
        f_val.write(str(entr) + "\n")
    f_val.write("Y_validate\n")
    for entr in y_validate:
        f_val.write(str(entr) + "\n")
    f_val.close()

    # Evaluate the models for testing
    df_predictions = XGBreg.predict(X_test)
    print('Ran model on observed set')

    timestep = [i for i in range(len(df_predictions))]
    observ_dict = {}
    color_indices = [-1]*590
    for i in range(590):
        if i in y_test.index:
            color_indices[i] = 0
        else:
            continue
        if(i > 280 and i < 514):
            color_indices[i] = 1
    ctr = 0
    for i in range(len(color_indices)):
        if(color_indices[i] != -1):
            observ_dict[i] = ctr
            ctr = ctr + 1

    df_pred_collection = []
    y_test_collection = []
    timestep_collection = []
    cur_y = []
    cur_df = []
    cur_time = []
    for i in range(len(color_indices)):
        if(color_indices[i] == -1):
            if(len(cur_y) == 0):
                continue
            df_pred_collection.append(cur_df)
            y_test_collection.append(cur_y)
            timestep_collection.append(cur_time)
            cur_y = []
            cur_df = []
            cur_time = []
            continue
        cur_y.append(y_test[i])
        cur_df.append(df_predictions[observ_dict[i]])
        cur_time.append(i)
    df_pred_collection.append(cur_df)
    y_test_collection.append(cur_y)
    timestep_collection.append(cur_time)

    print('Finised testing sorting')

    color_indices = list(filter(lambda a: a != -1, color_indices))
    colors = ["red", "blue"]
    colormap = clr.ListedColormap(colors)

    #test_outfile = "POST_P__test_res_800_3.txt"
    f_test = open(test_outfile, "w+")
    f_test.write("Testing predictions\n")
    for col_pred in df_pred_collection:
        for entr in col_pred:
            f_test.write(str(entr) + "\n")
    f_test.write("Y_test\n")
    for col_obs in y_test_collection:
        for entr in col_obs:
            f_test.write(str(entr) + "\n")
    f_test.write("Timestamp\n")
    for time_col in timestep_collection:
        for entr in time_col:
            f_test.write(str(entr) + "\n")
    f_test.close()


    feat_imp(XGBreg)



def main(argv):

    val_outfile = "VALIDATION_OUTPUT.txt"
    test_outfile = "TESTING_OUTPUT.txt"
    training_data_file = ""
    testing_data_file = ""
    validation_data_file = ""
    is_tune = False

    if(len(sys.argv) < 4 or len(sys.argv) > 9):
        usage()

    if(len(sys.argv) == 4):
        training_data_file = str(sys.argv[1])
        validation_data_file = str(sys.argv[2])
        testing_data_file = str(sys.argv[3])

    if(len(sys.argv) == 5):
        training_data_file = str(sys.argv[1])
        validation_data_file = str(sys.argv[2])
        testing_data_file = str(sys.argv[3])
        if(sys.argv[4] != '-h'):
            usage()
        is_tune = True

    if(len(sys.argv) == 6):
        if('-h' in sys.argv):
            usage()
        if(sys.argv[4] not in ['-ov','-ot']):
            usage()
        if(sys.argv[4] == '-ov'):
            val_outfile = sys.argv[4]
        else:
            test_outfile = sys.argv[4]

    if(len(sys.argv) == 8):
        if('-h' in sys.argv):
            usage()
        if(sys.argv[4] not in ['-ov','-ot'] or sys.argv[6] not in ['-ov','-ot']):
            usage()

        if(sys.argv[4] == '-ov'):
            val_outfile = sys.argv[4]
            test_outfile = sys.argv[6]
        else:
            test_outfile = sys.argv[4]
            val_outfile = sys.argv[6]

    if(len(sys.argv) == 7):
        if(sys.argv[4] != '-h'):
            usage()
        is_tune = True

        if(sys.argv[5] not in ['-ov','-ot']):
            usage()

        if(sys.argv[5] == '-ov'):
            val_outfile = sys.argv[5]
        else:
            test_outfile = sys.argv[5]

    if(len(sys.argv) == 9):
        if(sys.argv[4] != '-h'):
            usage()
        is_tune = True

        if(sys.argv[5] not in ['-ov','-ot'] or sys.argv[7] not in ['-ov','-ot']):
            usage()

        if(sys.argv[5] == '-ov'):
            val_outfile = sys.argv[5]
            test_outfile = sys.argv[7]
        else:
            test_outfile = sys.argv[5]
            val_outfile = sys.argv[7]
    
    vari = ["O3_SRF", "NO_SRF", "NOX_SRF", "CO_SRF",
            "C2H6_SRF", "ETH_SRF", "OLET_SRF", "PAR_SRF", "TOL_SRF", "XYL_SRF", "CH3OH_SRF",
            "ALD2_SRF", "AONE_SRF", "T", "RELHUM", "Mass_so4", "Mass_bc", "Mass_oa", "Mass_nh4", "Mass_no3"]
    

    #For Drop Test - Any feature under Threshold=4% share dropped
    #vari = ["O3_SRF", "NO_SRF", "NOX_SRF", "CO_SRF","C2H6_SRF", "ETH_SRF", "PAR_SRF", "CH3OH_SRF","T", "RELHUM", "Mass_so4", "Mass_bc", "Mass_oa", "Mass_nh4"]

    pred = "chi_abd"

    #training_data_file = "../data/train_v20_800.csv"
    #validation_data_file = "../data/test_v20_model_400.csv"
    #testing_data_file = "../data/test_fixed_v6.csv"

    df_test_ori = pd.read_csv(testing_data_file)

    df_validate_ori = pd.read_csv(validation_data_file)

    df_train_ori = pd.read_csv(training_data_file)

    ml_workflow_grid_search(df_train=df_train_ori.dropna(), df_validate=df_validate_ori.dropna(), df_test=df_test_ori.dropna(),
                            features_ls=vari, pred=pred, time_label=df_test_ori.dropna()["Time"], is_tune, val_outfile, test_outfile)


if __name__ == '__main__':
    main()
