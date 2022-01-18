import xgboost as xgb
from xgboost import XGBRegressor
from sklearn import model_selection, metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
#from sklearn.metrics import mean_absolute_percentage_error
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
    return 0.0123748445508538, 1088, 8
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


def mae(y_ori, y_pred):
    return mean_absolute_error(y_ori, y_pred)


def evs(y_ori, y_pred):
    return explained_variance_score(y_ori, y_pred)


def r2_f(y_ori, y_pred):
    return r2_score(y_ori, y_pred)


def mape(y_ori, y_pred):
    y_true, y_pred = np.array(y_ori), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Calculate squared_error


def squared_error(y_ori, y_pred):
    y_ori = np.array(y_ori)
    return np.sum((y_pred.reshape(len(y_pred), 1) - y_ori * (y_pred.reshape(len(y_pred), 1) - y_ori)))

# Calculate coefficient of determination


def coefficient_of_determination(y_ori, y_pred):
    """
    ref: https://en.wikipedia.org/wiki/Coefficient_of_determination
    """
    y_mean_pred = np.full(np.asarray(y_ori).shape, np.mean(y_ori))
    SSres = squared_error(y_ori, y_pred)
    SStot = squared_error(y_ori, y_mean_pred)
    return 1 - (SSres/SStot)

# Calculate index of agreement


def index_agreement(ori, pred):
    """
    ref: https://www.rforge.net/doc/packages/hydroGOF/d.html
    index of agreement
    input:
        pred: simulated
        ori: observed
    output:
        ia: index of agreement
    """
    ori = np.array(ori)
    pred = pred.reshape(len(pred), 1)
    ia = 1 - (np.sum((ori-pred)**2))/(np.sum(
        (np.abs(pred-np.mean(ori))+np.abs(ori-np.mean(ori)))**2))
    return ia

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


    #xgb.plot_importance(XGBreg)
    #plt.savefig("3_800_fscore.png")
    #plt.show()

def ml_workflow_grid_search(df_train, df_validate, df_test, features_ls, pred, para_dict, time_label):

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


    # Start grid search
    print("Start grid search")
    start_time = time.time()
    CV_xgb = GridSearchCV(XGBreg, param_grid=param_grid, verbose=2,cv=kfold, scoring='neg_mean_squared_error')
    CV_xgb.fit(X_train, y_train)
    print(CV_xgb.best_score_)
    print(CV_xgb.best_params_)
    print(CV_xgb.cv_results_)
    print("Finish grid search, it takes", time.time()-start_time)

    # Get the best combination 
    print('learning_rate',CV_xgb.best_params_['learning_rate'])
    print('max_depth',CV_xgb.best_params_['max_depth'])
    print('n_estimators',CV_xgb.best_params_['n_estimators'])
        '''
    learning_rate_bayes, n_estimators_bayes, max_depth_bayes = hyper_tune(
        X_train, y_train, n_iter=128)

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
    

    '''
    plt.figure(figsize=(10, 10))
    plt.scatter(y_train, df_predictions, s=0.8)
    plt.plot([0, 1], [0, 1], color="red")
    plt.title("Training ")
    plt.ylabel("Prediction chi_abd", fontsize=20)
    plt.xlabel("Reference chi_abd", fontsize=20)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.savefig("3_rand_200_hyperparam_4_train.png")
    plt.show()
    '''

    
    print('Testing (Training - PartMC) set: The coefficient of determination is:',
          "{0:.3f}".format(coefficient_of_determination(y_train, df_predictions)))
    print('Testing (Training - PartMC) set: The index of agreement is:',
          "{0:.3f}".format(index_agreement(y_train, df_predictions)))
    print('Testing (Training - PartMC) set: The root mean squared error is:',
          "{0:.3f}".format(np.sqrt(mean_squared_error(y_train, df_predictions))))
    print('Testing (Training - PartMC) set: The mean absolute error is:',
          "{0:.3f}".format(mae(y_train, df_predictions)))
    print('Testing (Training - PartMC) set: R2 is:',
          "{0:.3f}".format(r2_f(y_train, df_predictions)))
    print('Testing (Training - PartMC) set: The explained variance is:',
          "{0:.3f}".format(evs(y_train, df_predictions)))
    print('Testing (Training - PartMC) set: The mean absolute percentage error is:',
          "{0:.3f}".format(mape(y_train, df_predictions)))
    

    # Run model on validation set
    df_predictions = XGBreg.predict(X_validate)
    print('Ran model on validation set')

    
    f_val = open("3_validation_res_800_3.txt", "w+")
    f_val.write("Validation predictions\n")
    for entr in df_predictions:
        f_val.write(str(entr) + "\n")
    f_val.write("Y_validate\n")
    for entr in y_validate:
        f_val.write(str(entr) + "\n")
    f_val.close()
    

    '''
    plt.figure(figsize=(10, 10))
    plt.scatter(y_validate, df_predictions, s=0.8)
    plt.plot([0, 1], [0, 1], color="red")
    plt.title("Validation")
    plt.ylabel("Prediction chi_abd", fontsize=20)
    plt.xlabel("Reference chi_abd", fontsize=20)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.savefig("rand_200_hyperparam_4_val.png")
    #plt.show()
    '''

    print('Testing (Validation - PartMC) set: The coefficient of determination is:',
          "{0:.3f}".format(coefficient_of_determination(y_validate, df_predictions)))
    print('Testing (Validation - PartMC) set: The index of agreement is:',
          "{0:.3f}".format(index_agreement(y_validate, df_predictions)))
    print('Testing (Validation - PartMC) set: The root mean squared error is:',
          "{0:.3f}".format(np.sqrt(mean_squared_error(y_validate, df_predictions))))
    print('Testing (Validation - PartMC) set: The mean absolute error is:',
          "{0:.3f}".format(mae(y_validate, df_predictions)))
    print('Testing (Validation - PartMC) set: R2 is:',
          "{0:.3f}".format(r2_f(y_validate, df_predictions)))
    print('Testing (Validation - PartMC) set: The explained variance is:',
          "{0:.3f}".format(evs(y_validate, df_predictions)))
    print('Testing (Validation - PartMC) set: The mean absolute percentage error is:',
          "{0:.3f}".format(mape(y_validate, df_predictions)))


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

    f_test = open("POST_P__test_res_800_3.txt", "w+")
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

    '''
    plt.figure(figsize=(10, 10))
    for i in range(len(y_test_collection)):
        plt.plot(timestep_collection[i], y_test_collection[i], color="red")
        plt.plot(timestep_collection[i], df_pred_collection[i], color="blue")
    #plt.plot(timestep, y_test, color="red")
    #plt.plot(timestep, df_predictions, color="blue")
    plt.plot([281, 281], [0, 1], 'k-')
    plt.plot([514, 514], [0, 1], 'k-')
    #plt.figure(figsize=(10, 10))
    #plt.plot(timestep, y_test, color="red")
    #plt.plot(timestep, df_predictions, color="blue")
    plt.title("Testing values")
    plt.ylabel("Chi_abd", fontsize=10)
    plt.xlabel("Time (Hours since start)", fontsize=10)
    plt.ylim(0, 1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig("rand_200_hyperparam_4_comp.png")
    plt.show()
    

        
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test,df_predictions, s=0.8,c=color_indices, cmap=colormap)
    plt.plot([y_test.min(), y_test.max()], [
             y_test.min(), y_test.max()], color="blue")
    plt.title("Testing")
    plt.ylabel("Predicted chi_abd", fontsize=20)
    plt.xlabel("Actual chi_abd", fontsize=20)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.savefig("rand_800_hyperparam_4_predact.png")
    plt.show()
    '''

    print('Testing (Observed) set: The coefficient of determination is:',
          "{0:.3f}".format(coefficient_of_determination(y_test, df_predictions)))
    print('Testing (Observed) set: The index of agreement is:',
          "{0:.3f}".format(index_agreement(y_test, df_predictions)))
    print('Testing (Observed) set: The root mean squared error is:',
          "{0:.3f}".format(np.sqrt(mean_squared_error(y_test, df_predictions))))
    print('Testing (Observed) set: The mean absolute error is:',
          "{0:.3f}".format(mae(y_test, df_predictions)))
    print('Testing (Obeserved) set: R2 is:',
          "{0:.3f}".format(r2_f(y_test, df_predictions)))
    print('Testing (Observed) set: The explained variance is:',
          "{0:.3f}".format(evs(y_test, df_predictions)))
    print('Testing set: The mean absolute percentage error is:',
          "{0:.3f}".format(mape(y_test, df_predictions)))

    feat_imp(XGBreg)

    pickle.dump(XGBreg, open("./xgb_model/"+pred+".dat", "wb"))
    print("************************************")


def main():
    print("XGBoost version:", xgb.__version__)

    
    vari = ["O3_SRF", "NO_SRF", "NOX_SRF", "CO_SRF",
            "C2H6_SRF", "ETH_SRF", "OLET_SRF", "PAR_SRF", "TOL_SRF", "XYL_SRF", "CH3OH_SRF",
            "ALD2_SRF", "AONE_SRF", "T", "RELHUM", "Mass_so4", "Mass_bc", "Mass_oa", "Mass_nh4", "Mass_no3"]
    

    #For Drop Test - Any feature under Threshold=4% share dropped
    #vari = ["O3_SRF", "NO_SRF", "NOX_SRF", "CO_SRF","C2H6_SRF", "ETH_SRF", "PAR_SRF", "CH3OH_SRF","T", "RELHUM", "Mass_so4", "Mass_bc", "Mass_oa", "Mass_nh4"]

    pred = "chi_abd"

    df_test_ori = pd.read_csv("../data/test_fixed_v6.csv")
    # df_test.describe()

    df_validate_ori = pd.read_csv("../data/test_v20_model_400.csv")

    df_train_ori = pd.read_csv("../data/train_v20_800.csv")
    # df_train.describe()

    # Grid search space
    para_dict = {'learning_rate': [0.05],
                 'max_depth': [5],
                 'n_estimators': [256]}
    ml_workflow_grid_search(df_train=df_train_ori.dropna(), df_validate=df_validate_ori.dropna(), df_test=df_test_ori.dropna(),
                            features_ls=vari, pred=pred, para_dict=para_dict, time_label=df_test_ori.dropna()["Time"])


if __name__ == '__main__':
    main()
