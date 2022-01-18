from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error
import matplotlib.pyplot as plt 
import numpy as np

def evs(y_ori, y_pred):
    return explained_variance_score(y_ori, y_pred)


def r2_f(y_ori, y_pred):
    return r2_score(y_ori, y_pred)

def rmse(y_ori, y_pred):
    return mean_squared_error(y_ori, y_pred, squared=False)


def mape(y_ori, y_pred):
    y_true, y_pred = np.array(y_ori), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def display_metrics(results):
    print("Training Results\n========\n")
    r2_train = r2_f(results["train_act"], results["train_pred"])
    rmse_train = rmse(results["train_act"], results["train_pred"])
    mape_train = mape(results["train_act"], results["train_pred"])
    #print('R^2: ' + str(r2_train))
    print('RMSE: ' + str(rmse_train))
    print('MAPE: ' + str(mape_train) + '%\n')

    print("Validation Results\n========\n")
    r2_validate = r2_f(results["val_act"], results["val_pred"])
    rmse_validate = rmse(results["val_act"], results["val_pred"])
    mape_validate = mape(results["val_act"], results["val_pred"])
    #print('R^2: ' + str(r2_validate))
    print('RMSE: ' + str(rmse_validate))
    print('MAPE: ' + str(mape_validate) + '%\n')

    print("Testing Results\n========\n")
    r2_test= r2_f(results["test_act"], results["test_pred"])
    rmse_test = rmse(results["test_act"], results["test_pred"])
    mape_test = mape(results["test_act"], results["test_pred"])
    #print('R^2: ' + str(r2_test))
    print('RMSE: ' + str(rmse_test))
    print('MAPE: ' + str(mape_test) + '%\n')

def values_graph():
    f = open("values.txt","r")
    lines = f.readlines()
    res = dict()
    for line in lines:
        stuff = line.split()
        name = stuff[0]
        val = float(stuff[1])
        res[val] = name
    
    sorted_vals = list(reversed(sorted(res.keys())))
    labels_list = []
    for val in sorted_vals:
        labels_list.append(res[val])

    labels_list = ['O3', 'TEMP', 'RELHUM', 'CO', 'NO', 'BC', 'OA', 'NH4', 'PAR', 'NOX', 'ETH', 'CH3OH', 'C2H6', 'SO4', 'NO3', 'TOL', 'AONE', 'OLET', 'XYL', 'ALD2']
    plt.bar(labels_list,sorted_vals)
    #plt.pie(sorted_vals,labels=labels_list, autopct=lambda p: '{:.2f}%'.format(p * sum(sorted_vals) / 100),shadow=True)
    plt.ylabel("F-Score Share (Percentage)")
    plt.yticks(np.arange(0,13,1))
    plt.plot([-0.4,19.3], [4,4], 'k-')
    plt.xlabel("Species")
    plt.show()



def marine_testing(results):
    timestep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,62, 63, 64, 65, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 142, 143, 144, 145, 150, 151, 152, 153, 154, 155, 156, 157, 158, 161, 163, 165, 166, 168, 173, 175, 177, 178, 179, 181, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 197, 198, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 225, 226, 227, 228, 229, 230, 244, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 273, 274, 275, 276, 277, 278, 280, 284, 288, 289, 290, 291, 292, 294, 295, 296, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 407, 415, 416, 417, 418, 419, 420, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589]
    start = timestep.index(280)
    end = timestep.index(515) + 1
    results["test_act_marine"] = results["test_act"][start:end]
    results["test_pred_marine"] = results["test_pred"][start:end]

    print("Marine Testing Results\n========\n")
    r2_test= r2_f(results["test_act_marine"], results["test_pred_marine"])
    rmse_test = rmse(results["test_act_marine"], results["test_pred_marine"])
    mape_test = mape(results["test_act_marine"], results["test_pred_marine"])
    #print('R^2: ' + str(r2_test))
    print('RMSE: ' + str(rmse_test))
    print('MAPE: ' + str(mape_test) + '%\n')

def display_plots(results):
    timestep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,62, 63, 64, 65, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 142, 143, 144, 145, 150, 151, 152, 153, 154, 155, 156, 157, 158, 161, 163, 165, 166, 168, 173, 175, 177, 178, 179, 181, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 197, 198, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 225, 226, 227, 228, 229, 230, 244, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 273, 274, 275, 276, 277, 278, 280, 284, 288, 289, 290, 291, 292, 294, 295, 296, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 407, 415, 416, 417, 418, 419, 420, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589]
    
    #Training
    plt.figure(figsize=(10, 10))
    plt.scatter(results["train_act"], results["train_pred"], s=0.8)
    plt.plot([0, 1], [0, 1], color="red")
    plt.title("Training")
    plt.ylabel("Prediction chi_abd", fontsize=20)
    plt.xlabel("Reference chi_abd", fontsize=20)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.show()

    #Validation
    plt.figure(figsize=(10, 10))
    plt.scatter(results["val_act"], results["val_pred"], s=0.8)
    plt.plot([0, 1], [0, 1], color="red")
    #plt.title("Validation")
    plt.ylabel("Prediction chi_abd", fontsize=20)
    plt.xlabel("Reference chi_abd", fontsize=20)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.show()

    #Testing
    plt.figure(figsize=(10, 10))
    plt.plot(timestep, results["test_act"], color="red")
    plt.plot(timestep, results["test_pred"], color="blue")
    plt.plot([281, 281], [0, 1], 'k-')
    plt.plot([514, 514], [0, 1], 'k-')
    #plt.title("Testing values")
    plt.ylabel("Chi_abd", fontsize=20)
    plt.xlabel("Time", fontsize=20)
    locs, labels = plt.xticks()
    plt.xticks(locs, [labels[0],'15th Jan','20th Jan', '24th Jan', '28th Jan', '2nd Feb', '6th Feb', '11th Feb', labels[7]])
    plt.text(350, 0.95, 'Marine', fontsize=14)
    plt.text(0, 0.95, 'Continental', fontsize=14)
    plt.text(550, 0.95, 'Continental', fontsize=14)
    plt.ylim(0, 1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=20)
    plt.show()


def main():
    file_train = "paper_train_res_800.txt"
    file_val =  "paper_validation_res_800.txt"
    file_obs = "paper_test_res_800.txt"

    train_str = "Training predictions\n"
    val_str = "Validation predictions\n"
    test_str = "Testing predictions\n"
    y_train_str = "Y_train\n"
    y_val_str = "Y_validate\n"
    y_test_str = "Y_test\n"

    results = dict()
    results["train_pred"] = []
    results["train_act"] = []
    results["val_pred"] = []
    results["val_act"] = []
    results["test_pred"] = []
    results["test_act"] = []

    for fname in [file_train, file_val, file_obs]:
        if fname == file_train:
            pred = "train_pred"
            pred_str = train_str
            act = "train_act"
            act_str = y_train_str
        if fname == file_val:
            pred = "val_pred"
            pred_str = val_str
            act = "val_act"
            act_str = y_val_str
        if fname == file_obs:
            pred = "test_pred"
            pred_str = test_str
            act = "test_act"
            act_str = y_test_str

        f = open(fname, "r+")
        lines = f.readlines()
        
        is_pred = False
        is_act = False

        for line in lines:
            if(line == pred_str):
                is_pred = True
                is_act = False
                continue
            if(line == act_str):
                is_pred = False
                is_act = True
                continue

            if(is_pred):
                results[pred].append(float(line))
            if(is_act):
                results[act].append(float(line))
        is_pred = False
        is_act = False

    display_metrics(results)
    display_plots(results)

    #marine_testing(results)
    #values_graph()

if __name__ == '__main__':
    main()