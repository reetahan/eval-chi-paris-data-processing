from matplotlib import pyplot as plt
from matplotlib import colors as clrs
from matplotlib import cm
import pickle
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error
import numpy as np
import xlrd
import pandas as pd
import copy
import datetime
import sys
from sklearn.neighbors import KernelDensity

def validation_heatmaps(results_1,results_2,sname):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(10)
    fig.set_figwidth(20)

    the_range = np.array([[0,100],[0,100]])
    colors = ["#c8dbec", "#9fc5e8", "#7db2e0", "#6fa8dc","#559ddc","#4d98da", "#3d85c6", "#236cad"]
    cmap= clrs.ListedColormap(colors)
    

    label = "(a)"
    print("Validation Results\n========\n")
    rmse_test = rmse(results_1["val_act"], results_1["val_pred"])
    mape_test = mape(results_1["val_act"], results_1["val_pred"])
    rmse_str = 'RMSE: ' + str(round(rmse_test,2)) + "%\n"
    mape_str = 'MAPE: ' + str(round(mape_test,2)) + "%"
    print(rmse_str)
    print(mape_str)

    
    h,xedges,yedges,img = ax1.hist2d(results_1["val_act"], results_1["val_pred"], bins=(100, 100), range=the_range, cmap=cmap,cmin=0.9)
    bounds = []
    interval = int((np.nanmax(h) - 1) / (len(colors) - 1))
    for i in range(1,int(np.nanmax(h)),interval):
        bounds.append(i)
    norm = clrs.BoundaryNorm(bounds, cmap.N)
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),label="Number of Points", ax=ax1, orientation="horizontal",fraction=0.041, pad=0.13)
    
   
    plt.rcParams['text.usetex'] = True
    ax1.set_ylabel('Predicted Mixing State Index ' + r"$\chi_{\rm pred}$" + ' / %')
    ax1.set_xlabel('Mixing State Index ' + r"$\chi_{\rm ref}$" + ' / %')
    plt.rcParams['text.usetex'] = False
    ax1.text(0.05, 0.05, rmse_str + mape_str, transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.text(0.05, 0.9, label, transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True)



    label = "(b)"
    
    print("Validation Results\n========\n")
    rmse_test = rmse(results_2["val_act"], results_2["val_pred"])
    mape_test = mape(results_2["val_act"], results_2["val_pred"])
    rmse_str = 'RMSE: ' + str(round(rmse_test,2)) + "%\n"
    mape_str = 'MAPE: ' + str(round(mape_test,2)) + "%"
    print(rmse_str)
    print(mape_str)

    #colors = ["#ffffff", "#9fc5e8", "#6fa8dc", "#4d98da", "#1b65a6", "#073763","#133a5d","#010f1c"]
    colors = ["#9fc5e8", "#7db2e0","#559ddc","#4d98da", "#3d85c6", "#1b65a6", "#0b5394","#073763","#133a5d","#010f1c"]
    cmap = clrs.ListedColormap(colors)

    h,xedges,yedges,img = ax2.hist2d(results_2["val_act"], results_2["val_pred"], bins=(100, 100), range=the_range, cmap=cmap,cmin=0.9)

    bounds = []
    interval = int((np.nanmax(h) - 1) / (len(colors) - 1))
    for i in range(1,int(np.nanmax(h)),interval):
        bounds.append(i)
    norm = clrs.BoundaryNorm(bounds, cmap.N)
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),label="Number of Points", ax=ax2, orientation="horizontal",fraction=0.041, pad=0.13)
   

    plt.rcParams['text.usetex'] = True
    ax2.set_ylabel('Predicted Mixing State Index ' + r"$\chi_{\rm pred}$" + ' / %')
    ax2.set_xlabel('Mixing State Index ' + r"$\chi_{\rm ref}$" + ' / %')
    plt.rcParams['text.usetex'] = False
    ax2.text(0.05, 0.05, rmse_str + mape_str, transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.text(0.05, 0.9, label, transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True)


    plt.savefig(sname,bbox_inches='tight', dpi=275)
    plt.show()


def rmse(y_ori, y_pred):
    return mean_squared_error(y_ori, y_pred, squared=False)


def mape(y_ori, y_pred):
    y_true, y_pred = np.array(y_ori), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def parse_aerosols(aerosol_raw):
    aerosol = aerosol_raw.sheet_by_index(0)
    res = dict()
    skipped = set()
    for i in range(1, aerosol.nrows):
        key_str_raw = aerosol.cell_value(i, 0)
        key_str = str(datetime.datetime(
            *xlrd.xldate_as_tuple(key_str_raw, aerosol_raw.datemode)))

        to_skip = False
        for j in range(aerosol.ncols):
            if(aerosol.cell_type(i, j) in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK)):
                to_skip = True
                break
        if(to_skip):
            skipped.add(key_str)
            continue

        BC = float(aerosol.cell_value(i, 1))
        OA = float(aerosol.cell_value(i, 2))
        NH4 = float(aerosol.cell_value(i, 3))
        NO3 = float(aerosol.cell_value(i, 4))
        SO4 = float(aerosol.cell_value(i, 5))
        TotalMassBulk = float(aerosol.cell_value(i, 6))
        MFBC = float(aerosol.cell_value(i, 7))
        MFOA = float(aerosol.cell_value(i, 8))
        MFNH4 = float(aerosol.cell_value(i, 9))
        MFNO3 = float(aerosol.cell_value(i, 10))
        MFSO4 = float(aerosol.cell_value(i, 11))
        D_alpha = float(aerosol.cell_value(i, 12))
        D_gamma = float(aerosol.cell_value(i, 13))
        Chi = float(aerosol.cell_value(i, 14)) / 100.0

        val_dict = {'Mass_bc': BC, 'Mass_oa': OA, 'Mass_nh4': NH4, 'Mass_no3': NO3, 'Mass_so4': SO4, 'TotalMassBulk': TotalMassBulk,
                    'MFBC': MFBC, 'MFOA': MFOA, 'MFNH4': MFNH4, 'MFNO3': MFNO3, 'MFSO4': MFSO4, 'D_alpha': D_alpha, 'D_gamma': D_gamma, 'chi_abd': Chi}
        res[key_str] = val_dict

    return res, skipped

def get_times(miss):

    timestep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 62, 63, 64, 65, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 142, 143, 144, 145, 150, 151, 152, 153, 154, 155, 156, 157, 158, 161, 163, 165, 166, 168, 173, 175, 177, 178, 179, 181, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 197, 198, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 225, 226, 227, 228, 229, 230, 244, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 273, 274, 275, 276, 277, 278, 280, 284, 288, 289, 290, 291, 292, 294, 295, 296, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342,
        343, 344, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 407, 415, 416, 417, 418, 419, 420, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589]
    if(miss):
        timestep = list(np.arange(590))
    aerosols_xl_file = "/home/reetahan/research/atms490/fall20/init-analysis/mass_conc.xlsx"
    apd = pd.read_excel(aerosols_xl_file)
    aerosols_xl = xlrd.open_workbook(aerosols_xl_file)
    aerosol_res, skipped = parse_aerosols(aerosols_xl)
    chi_list = []
    for date in aerosol_res:
        chi_list.append(aerosol_res[date]['chi_abd']*100)
    obs_timestep = np.arange(0, 591)
    chi_list = chi_list[19:len(chi_list)]
    df_test_ori = pd.read_csv("../summer21/test_fixed_v6.csv")
    y_ep = df_test_ori["chi_abd"] * 100
    upd = df_test_ori.dropna()
    y_ep_2 = upd["chi_abd"] * 100


    timestamps = list(apd["DateAndTime(Local)"])
    for i in range(len(timestamps)):
        timestamps[i] = str(timestamps[i])
    
    raw_observed = list(apd["Chi"])
    cleansed_timestamps = list(df_test_ori["Time"])
    double_cleansed_timestamps = []
    for i in range(len(timestep)):
        double_cleansed_timestamps.append(cleansed_timestamps[timestep[i]])

    ct = 0
    for ts in timestamps:
        if(ts not in cleansed_timestamps):
            ct = ct + 1

    original_timestep = np.arange(0, len(timestamps))
    new_timestep = []
    for i in range(len(timestamps)):
        if(miss):
            if(timestamps[i] in cleansed_timestamps):
                new_timestep.append(i)
        else:
            if(timestamps[i] in double_cleansed_timestamps):
                new_timestep.append(i)

    return timestamps, original_timestep, new_timestep

def tseries_plots_nonprim(total_results,miss,marine,obs_loc,sname,count):

    
    fig, axs = plt.subplots(count)
    fig.set_figheight(58)
    fig.set_figwidth(45.18)
    plt.xlabel("Date")
    locs, labels = plt.xticks()

    new_locs = []
    for i in range(2, 28, 5):
        new_locs.append(i*24)
    plt.xticks(new_locs, ['17th Jan', '22th Jan',
               '27nd Jan', '1st Feb', '6th Feb', '11th Feb'])
    
    
    plt.xticks()
    letters = ['(a)', '(b)', '(c)']

    for i in range(count):
        axs[i].grid()
        axs[i].set_ylim(20, 80)

        plt.rcParams['text.usetex'] = True
        axs[i].set_ylabel('Mixing State Index ' + r'$\chi$' + ' / %')
        plt.rcParams['text.usetex'] = False

        if(marine[i]):
            new_locs = []
            for n in range(0,11,2):
                new_locs.append(n*24)
            axs[i].set_xticks(new_locs, ['28th Jan', '30th Jan',
                       '1st Feb', '3rd Feb', '5th Feb', '7th Feb'])

        first = True

        for results in total_results[i]:
            timestamps, original_timestep, new_timestep = get_times(miss[i][results])
            cleansed_observed = []
            predictions = []
            predictions_cont = []
            cleansed_observed_cont = []

            j = 0
            for m in range(len(timestamps)):
                if(m in new_timestep):
                    cleansed_observed.append(total_results[i][results]["test_act"][j])
                    predictions.append(total_results[i][results]["test_pred"][j])
                    predictions_cont.append(total_results[i][results]["test_pred"][j])
                    cleansed_observed_cont.append(total_results[i][results]["test_act"][j])
                    j = j + 1
                else:
                    cleansed_observed.append(np.NaN)
                    predictions.append(np.NaN)
                    cleansed_observed_cont.append(np.NaN)
                    predictions_cont.append(np.NaN)


            # Testing
            if(marine[i]):
                cont_ts = list(original_timestep[:314]) + list(original_timestep[555:])
                cleansed_observed_cont = list(cleansed_observed[:314]) + list(cleansed_observed[555:])
                predictions_cont = list(predictions[:314]) + list(predictions[555:])
                original_timestep = original_timestep[314:555]
                original_timestep = [x - original_timestep[0] for x in original_timestep]
                cleansed_observed= cleansed_observed[314:555]
                predictions = predictions[314:555]

            if(first):
                axs[i].plot(original_timestep, cleansed_observed,color='blue', label='Observed',linewidth=4.4)
                axs[i].annotate("Observed", xycoords='axes fraction',xy=obs_loc[i][0], xytext=obs_loc[i][1],
                color='blue',textcoords='axes fraction',arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
                first = False

            axs[i].plot(original_timestep, predictions, color=total_results[i][results]["color"], 
                label=total_results[i][results]["label"],linewidth=4.4)
            axs[i].annotate(total_results[i][results]["label"], xycoords='axes fraction',xy=total_results[i][results]["label_loc"][0], 
                xytext=total_results[i][results]["label_loc"][1],color= total_results[i][results]["color"],
                textcoords='axes fraction', arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))

            nan_idxs = np.argwhere(np.isnan(predictions))
            predictions = [x for x in predictions if np.isnan(x) == False]
            cleansed_observed = np.delete(cleansed_observed, nan_idxs)
            nan_idxs_cont = np.argwhere(np.isnan(predictions_cont))
            predictions_cont = [x for x in predictions_cont if np.isnan(x) == False]
            cleansed_observed_cont = np.delete(cleansed_observed_cont, nan_idxs_cont)

            print(results)
            print("Results\n========\n")
            if(marine[i]):
                print('Marine')
            if(miss):
                print('Miss')
            rmse_test = round(rmse(predictions, cleansed_observed),2)
            mape_test = round(mape(predictions, cleansed_observed),2)

            rmse_str = 'RMSE: ' + str(rmse_test) + "%\n"
            mape_str = 'MAPE: ' + str(mape_test) + "%"
            print(rmse_str)
            print(mape_str)
            plt.text(0.02, 0.9, letters[i], transform=axs[i].transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=1.0))

           
            print('Just Continental')
            rmse_test = rmse(predictions_cont, cleansed_observed_cont)
            mape_test = mape(predictions_cont, cleansed_observed_cont)

            print('RMSE: ' + str(rmse_test))
            print('MAPE: ' + str(mape_test) + '%\n')

            if(i != (count -1)):
                plt.setp(axs[i].get_xticklabels(), visible=False)

  
    plt.savefig(sname,bbox_inches='tight',dpi=275)
    plt.show()

def tseries_plots_prim(total_results,miss,obs_loc,sname):

    rmses = []
    mapes = []
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(16.18)
    plt.grid()
    plt.xlabel("Date")
    plt.rcParams['text.usetex'] = True
    plt.ylabel('Mixing State Index ' + r'$\chi$' + ' / %')
    plt.rcParams['text.usetex'] = False
    locs, labels = plt.xticks()

    new_locs = []
    for i in range(2, 28, 5):
        new_locs.append(i*24)
    plt.xticks(new_locs, ['17th Jan', '22th Jan',
               '27nd Jan', '1st Feb', '6th Feb', '11th Feb'])
    
    plt.ylim(20, 80)
    plt.xticks()
    plt.yticks()

    first = True
    for results in total_results:
        timestamps, original_timestep, new_timestep = get_times(miss[results])
        cleansed_observed = []
        predictions = []
        predictions_cont = []
        cleansed_observed_cont = []

        j = 0
        for i in range(len(timestamps)):
            if(i in new_timestep):
                cleansed_observed.append(total_results[results]["test_act"][j])
                predictions.append(total_results[results]["test_pred"][j])
                predictions_cont.append(total_results[results]["test_pred"][j])
                cleansed_observed_cont.append(total_results[results]["test_act"][j])
                j = j + 1
            else:
                cleansed_observed.append(np.NaN)
                predictions.append(np.NaN)
                cleansed_observed_cont.append(np.NaN)
                predictions_cont.append(np.NaN)


        if(first):
            plt.plot(original_timestep, cleansed_observed,color='blue', label='Observed')
            plt.annotate("Observed", xycoords='axes fraction',xy=obs_loc[0], xytext=obs_loc[1],
            color='blue',textcoords='axes fraction',arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
            first = False
        plt.plot(original_timestep, predictions, color=total_results[results]["color"], 
            label=total_results[results]["label"])
        plt.annotate(total_results[results]["label"], xycoords='axes fraction',xy=total_results[results]["label_loc"][0], 
            xytext=total_results[results]["label_loc"][1],color= total_results[results]["color"],
            textcoords='axes fraction', arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))


        cont_ts = list(original_timestep[:314]) + list(original_timestep[555:])
        cleansed_observed_cont = list(cleansed_observed[:314]) + list(cleansed_observed[555:])
        predictions_cont = list(predictions[:314]) + list(predictions[555:])
        original_timestep = original_timestep[314:555]
        original_timestep = [x - original_timestep[0] for x in original_timestep]
        cleansed_observed= cleansed_observed[314:555]
        predictions = predictions[314:555]


        nan_idxs = np.argwhere(np.isnan(predictions))
        predictions = [x for x in predictions if np.isnan(x) == False]
        cleansed_observed = np.delete(cleansed_observed, nan_idxs)
        nan_idxs_cont = np.argwhere(np.isnan(predictions_cont))
        predictions_cont = [x for x in predictions_cont if np.isnan(x) == False]
        cleansed_observed_cont = np.delete(cleansed_observed_cont, nan_idxs_cont)

        print("Results\n========\n")

        rmse_test = round(rmse(predictions, cleansed_observed),2)
        mape_test = round(mape(predictions, cleansed_observed),2)
        
        rmses.append(rmse_test)
        mapes.append(mape_test)

        #print('Just Continental')
        rmse_test = rmse(predictions_cont, cleansed_observed_cont)
        mape_test = mape(predictions_cont, cleansed_observed_cont)
        print('RMSE: ' + str(rmse_test))
        print('MAPE: ' + str(mape_test) + '%\n')

    plt.axvline(x=314)
    plt.axvline(x=515)

    rmse_str = 'RMSE (XGB): ' + str(round(rmses[0],2)) + "%\n" + 'RMSE (AML): ' + str(round(rmses[0],2)) + "%\n"
    mape_str = 'MAPE (XGB): ' + str(round(mapes[0],2)) + "%\n" + 'MAPE (AML): ' + str(round(mapes[1],2)) + "%"
    print(rmse_str)
    print(mape_str)
    #plt.text(0.02, 0.05, rmse_str + mape_str, transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=1))

    plt.annotate("Continental", xycoords='axes fraction', xy=(0.15,1.05),color='purple')
    plt.annotate("Marine", xycoords='axes fraction', xy=(0.55,1.05),color='darkgreen')
    plt.annotate("Continental", xycoords='axes fraction', xy=(0.8,1.05),color='purple')
    plt.savefig(sname,bbox_inches='tight', dpi=275)
    plt.show()

def kdes(total_results, marine, kde_obs_loc,sname, miss):

    plt.figure(figsize=(16.18, 10))
    plt.ylabel("Probability Density")

    kde_res = {}
    legend_list = []
    color_list = []


    for result in total_results:
        for section in total_results[result]:

            timestamps, original_timestep, new_timestep = get_times(miss[result])
            cleansed_observed = []
            predictions = []
            predictions_cont = []
            cleansed_observed_cont = []
            j = 0
            for i in range(len(timestamps)):
                if(i in new_timestep):
                    cleansed_observed.append(total_results[result]["test_act"][j])
                    predictions.append(total_results[result]["test_pred"][j])
                    predictions_cont.append(total_results[result]["test_pred"][j])
                    cleansed_observed_cont.append(total_results[result]["test_act"][j])
                    j = j + 1
                else:
                    cleansed_observed.append(np.NaN)
                    predictions.append(np.NaN)
                    cleansed_observed_cont.append(np.NaN)
                    predictions_cont.append(np.NaN)

            if(section == "test_act"):
                kde_res["observed"] = np.array(cleansed_observed)
                if(marine):
                    kde_res["observed"] = np.array(cleansed_observed[314:555])
                nan_idxs = np.argwhere(np.isnan(kde_res["observed"]))
                kde_res["observed"] = np.delete(kde_res["observed"], nan_idxs)

                legend_list.append("Observed")
                color_list.append("blue")
            if(section == "test_pred"):

                legend_list.append(total_results[result]["label"])
                color_list.append(total_results[result]["color"])
                kde_res[total_results[result]["label"]] = np.array(predictions)
                if(marine):
                    kde_res[total_results[result]["label"]] = np.array(predictions[314:555])
                
                nan_idxs = np.argwhere(np.isnan(kde_res[total_results[result]["label"]]))
                kde_res[total_results[result]["label"]] = np.delete(kde_res[total_results[result]["label"]], nan_idxs)
               
            if(section == "test_act"):
                plt.annotate("Observed", xycoords='axes fraction',xy=kde_obs_loc[0], xytext=kde_obs_loc[1],
                color='blue',textcoords='axes fraction',arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
                first = False
            else:
                plt.annotate(total_results[result]["label"], xycoords='axes fraction',xy=total_results[result]["kde_label_loc"][0], 
                    xytext=total_results[result]["kde_label_loc"][1],color= total_results[result]["color"],
                    textcoords='axes fraction', arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))

    color_idx = 0

    for res in kde_res:

        cur_kde = KernelDensity(kernel='gaussian',bandwidth=2.5)
        cur_kde.fit(np.array(kde_res[res]).reshape(-1,1))


        vals_to_check = np.linspace(np.min(kde_res[res]),np.max(kde_res[res]),1000).reshape(-1, 1)
        x_axis = vals_to_check.flatten()
        scores= cur_kde.score_samples(vals_to_check)

        plt.fill_between(x_axis,np.exp(scores),alpha=0.7,color=color_list[color_idx])
        color_idx = color_idx + 1


    plt.grid()
    plt.xlabel('Mixing State Index ' + r'$\chi$' + ' / %')    
    plt.savefig(sname,bbox_inches='tight', dpi=275)
    plt.show()

def main():

    final_results = {}
    with open("NEW_PAPER_RESULTS.pkl","rb") as pickle_file:
        final_results = pickle.load(pickle_file)
    print(final_results.keys())




    # Section 3.2
    plt.rcParams.update({'font.size': 24})
    total_results = {}
    miss = {}
    total_results["missingincluded"] = final_results["missingincluded"]
    total_results["missingincluded"]["color"] = "orange"
    total_results["missingincluded"]["label"] = "XGBoost"
    total_results["missingincluded"]["label_loc"] = [(0.66,0.79),(0.61,0.9)]
    total_results["missingincluded"]["kde_label_loc"] =  [(0.4,0.60),(0.28,0.69)]
    total_results["automl_missingincluded"] = final_results["automl_missingincluded"]
    total_results["automl_missingincluded"]["color"] = "gray"
    total_results["automl_missingincluded"]["label"] = "AutoML"
    total_results["automl_missingincluded"]["label_loc"] = [(0.39,0.92),(0.24,0.92)]
    total_results["automl_missingincluded"]["kde_label_loc"] = [(0.46,0.73),(0.30,0.8)]
    obs_loc_ts = [(0.55,0.85),(0.41,0.92)]
    kde_obs_loc_ts = [(0.22,0.34),(0.12,0.45)]
    miss["missingincluded"] = True
    miss["automl_missingincluded"] =  True

    validation_heatmaps(total_results["missingincluded"],total_results["automl_missingincluded"],"4_test_1000edit.pdf")
    tseries_plots_prim(total_results,miss,obs_loc_ts,"4_prim_tseries_1000edit.pdf")
    kdes(total_results, True, kde_obs_loc_ts,"4_prim_kdes_1000edit.pdf",miss)

    miss_arr = []
    marine_arr = []
    obs_loc_arr = []
    total_total_results = []
    

    plt.rcParams.update({'font.size': 53})


    # Section 3.3
    total_results = {}
    miss = {}
    total_results["missingincluded"] = copy.deepcopy(final_results["missingincluded"])
    total_results["missingincluded"]["color"] = "orange"
    total_results["missingincluded"]["label"] = "Reference Prediction"
    total_results["missingincluded"]["label_loc"] = [(0.73,0.8),(0.78,0.88)]
    total_results["xgbmissing_const"] = final_results["xgbmissing_const"]
    total_results["xgbmissing_const"]["color"] = "gray"
    total_results["xgbmissing_const"]["label"] = "Constant Temp"
    total_results["xgbmissing_const"]["label_loc"] = [(0.63,0.83),(0.66,0.94)]
    miss["missingincluded"] = True
    miss["xgbmissing_const"] = True
    obs_loc_ts = [(0.31,0.83),(0.38,0.88)]

    total_total_results.append(total_results)
    miss_arr.append(miss)
    marine_arr.append(True)
    obs_loc_arr.append(obs_loc_ts)

    total_results = {}
    miss = {}
    total_results["xgbmissingincluded_200"] = final_results["xgbmissingincluded_200"]
    total_results["xgbmissingincluded_200"]["color"] = "purple"
    total_results["xgbmissingincluded_200"]["label"] = "N=200"
    total_results["xgbmissingincluded_200"]["label_loc"] = [(0.62,0.33),(0.7,0.17)]
    total_results["xgbmissingincluded_200"]["kde_label_loc"] = [(0.3,0.75),(0.1,0.9)]

    total_results["xgbmissingincluded_600"] = final_results["xgbmissingincluded_600"]
    total_results["xgbmissingincluded_600"]["color"] = "gray"
    total_results["xgbmissingincluded_600"]["label"] = "N=600"
    total_results["xgbmissingincluded_600"]["label_loc"] = [(0.52,0.67),(0.55,0.95)]
    total_results["xgbmissingincluded_600"]["kde_label_loc"] = [(0.38,0.84),(0.25,0.95)]

    total_results["xgbmissingincluded"] = final_results["missingincluded"]
    total_results["xgbmissingincluded"]["color"] = "orange"
    total_results["xgbmissingincluded"]["label"] = "Reference Prediction"
    total_results["xgbmissingincluded"]["label_loc"] = [(0.8,0.33),(0.75,0.1)]
    total_results["xgbmissingincluded"]["kde_label_loc"] = [(0.59,0.85),(0.63,0.95)]

    miss["xgbmissingincluded_200"] = True
    miss["xgbmissingincluded_600"] = True
    miss["xgbmissingincluded"] = True
    obs_loc_ts = [(0.46,0.38),(0.39,0.17)]
    kde_obs_loc_ts = [(0.78,0.85),(0.7,0.95)]

    total_total_results.append(total_results)
    miss_arr.append(miss)
    marine_arr.append(True)
    obs_loc_arr.append(obs_loc_ts)

    total_results = {}
    miss = {}
    total_results["missingincluded"] = copy.deepcopy(final_results["missingincluded"])
    total_results["missingincluded"]["color"] = "orange"
    total_results["missingincluded"]["label"] = "Reference Prediction"
    total_results["missingincluded"]["label_loc"] = [(0.69,0.76),(0.49,0.95)]

    total_results["xgbmissing9"] = final_results["xgbmissing9"]
    total_results["xgbmissing9"]["color"] = "gray"
    total_results["xgbmissing9"]["label"] = "VOC Features Dropped"
    total_results["xgbmissing9"]["label_loc"] = [(0.87,0.24),(0.72,0.02)]

    total_results["xgbmissingextreme"] = final_results["xgbmissingextreme"]
    total_results["xgbmissingextreme"]["color"] = "purple"
    total_results["xgbmissingextreme"]["label"] = "Only Top 4 Features Kept"
    total_results["xgbmissingextreme"]["label_loc"] = [(0.81,0.83),(0.73,0.95)]

    miss["missingincluded"] = True
    miss["xgbmissing9"] = True
    miss["xgbmissingextreme"] = True
    obs_loc_ts = [(0.32,0.82),(0.38,0.93)]

    total_total_results.append(total_results)
    miss_arr.append(miss)
    marine_arr.append(True)
    obs_loc_arr.append(obs_loc_ts)

    sname = "4_sensitivity_analysis_1000edit.pdf"
    count = 3
    tseries_plots_nonprim(total_total_results,miss_arr,marine_arr,obs_loc_arr,sname,count)



if __name__ == '__main__':
    main()