import xlrd
import numpy as np
import datetime
import pandas as pd
import sys
import getopt


def main():
    outfile = "OBSERVATION_PROCESSING_OUTPUT.csv"
    if(len(sys.argv) == 2 or len(sys.argv) > 3):
        usage()
    if(len(sys.argv) == 2):
        if(sys.argv[1] != '-o'):
            usage()
        outfile = str(sys.argv[2])

    aerosols_txt_file = "data/O3-CO-NO_LHVP_LSCE_winter_20101027_R1.ames.txt"
    time_rh_file = "data/megapoli_winter_temp_rh.xlsx"
    voc_file = "data/MEGAPOLI_VOCs_20200828.xlsx"
    aerosols_xl_file = "data/mass_conc.xlsx"
    

    txt_lines = []
    with open(aerosols_txt_file, "r") as f:
        for line in f:
            txt_lines.append(line)
    time_rh = xlrd.open_workbook(time_rh_file)
    voc = xlrd.open_workbook(voc_file)
    aerosols_xl = xlrd.open_workbook(aerosols_xl_file)

    txt_res = parse_text(txt_lines)
    time_rh_res = parse_timerh(time_rh)
    voc_res = parse_voc(voc)
    aerosol_res, skipped = parse_aerosols(aerosols_xl)
    write_result(txt_res, time_rh_res, voc_res, aerosol_res, outfile)

def usage():
    usage_str = 'Usage: python observ_processing.py [-o output_filename]'
    print(usage_str)
    sys.exit()

def parse_text(txt):
    res = dict()
    started = False
    cur_o3 = create_numpy_array()
    cur_co = create_numpy_array()
    cur_no = create_numpy_array()
    cur_nox = create_numpy_array()
    last_hr = -1
    minute_mark = 625

    for line in txt:
        if(len(line) >= 10 and line[0:10] == "Time (UTC)"):
            started = True
            continue
        if(not started):
            continue

        line = line.replace(",", ".")
        pieces = line.split()
        time = pieces[0]
        day = int(time[0:2])
        hr = int(time[3:6])
        o3 = float(pieces[1])
        co = float(pieces[2])
        no = float(pieces[3])
        nox = float(pieces[4])

        if(hr > ((minute_mark + 125/3) % 1000) or hr < last_hr):
            minute_mark = (minute_mark + 125/3) % 1000
            hour = int(hr/1000*24)
            avg_o3 = np.average(cur_o3)
            avg_co = np.average(cur_co)
            avg_no = np.average(cur_no)
            avg_nox = np.average(cur_nox)

            key_str = ""
            if(day < 32):
                if(hour < 10):
                    key_str = "2010-01-" + \
                        str(day) + " 0" + str(hour) + ":00:00"
                else:
                    key_str = "2010-01-" + \
                        str(day) + " " + str(hour) + ":00:00"
            else:
                day = day % 31
                if(day < 10):
                    if(hour < 10):
                        key_str = "2010-02-0" + \
                            str(day) + " 0" + str(hour) + ":00:00"
                    else:
                        key_str = "2010-02-0" + \
                            str(day) + " " + str(hour) + ":00:00"
                else:
                    if(hour < 10):
                        key_str = "2010-02-" + \
                            str(day) + " 0" + str(hour) + ":00:00"
                    else:
                        key_str = "2010-02-" + \
                            str(day) + " " + str(hour) + ":00:00"

            val_dict = {"O3_SRF": avg_o3, "CO_SRF": avg_co,
                        "NO_SRF": avg_no, "NOX_SRF": avg_nox}

            res[key_str] = val_dict

            cur_o3 = create_numpy_array()
            cur_co = create_numpy_array()
            cur_no = create_numpy_array()
            cur_nox = create_numpy_array()

        if(o3 != 9999):
            cur_o3 = np.append(cur_o3, o3)
        if(co != 9999):
            cur_co = np.append(cur_co, co)
        if(no != 9999):
            cur_no = np.append(cur_no, no)
        if(nox != 9999):
            cur_nox = np.append(cur_nox, nox)

        last_hr = hr

    return res


def parse_timerh(time_rh_raw):
    res = dict()
    time_rh = time_rh_raw.sheet_by_index(0)
    for i in range(1, time_rh.nrows):
        key_str_raw = time_rh.cell_value(i, 0)
        key_str = str(datetime.datetime(
            *xlrd.xldate_as_tuple(key_str_raw, time_rh_raw.datemode)))
        temp = float(time_rh.cell_value(i, 1))
        rh = float(time_rh.cell_value(i, 2))
        val_dict = {'T': temp, 'RELHUM': rh}
        res[key_str] = val_dict
    return res


def parse_voc(voc_raw):
    voc = voc_raw.sheet_by_index(1)
    res = dict()
    for i in range(2, voc.nrows):
        key_str_raw = voc.cell_value(i-1, 0)
        key_str = str(datetime.datetime(
            *xlrd.xldate_as_tuple(key_str_raw, voc_raw.datemode)))

        C2H6 = np.nan if voc.cell_value(
            i, 1) == '' else float(voc.cell_value(i, 1))
        ETH = np.nan if voc.cell_value(
            i, 2) == '' else float(voc.cell_value(i, 2))
        PAR3 = np.nan if voc.cell_value(
            i, 3) == '' else float(voc.cell_value(i, 3))
        OLET_PAR = np.nan if voc.cell_value(
            i, 4) == '' else float(voc.cell_value(i, 4))
        PAR4 = np.nan if voc.cell_value(
            i, 6) == '' else float(voc.cell_value(i, 6))
        PAR5 = np.nan if voc.cell_value(
            i, 9) == '' else float(voc.cell_value(i, 9))
        PAR6 = np.nan if voc.cell_value(
            i, 10) == '' else float(voc.cell_value(i, 10))
        TOL = np.nan if voc.cell_value(
            i, 11) == '' else float(voc.cell_value(i, 11))
        XYL = np.nan if voc.cell_value(
            i, 13) == '' else float(voc.cell_value(i, 13))
        CH3OH = np.nan if voc.cell_value(
            i, 14) == '' else float(voc.cell_value(i, 14))
        ALD2 = np.nan if voc.cell_value(
            i, 16) == '' else float(voc.cell_value(i, 16))
        AONE = np.nan if voc.cell_value(
            i, 17) == '' else float(voc.cell_value(i, 17))
        PAR = 3*PAR3 + 4*PAR4 + 5*PAR5 + 6*PAR6 + OLET_PAR
        OLET = OLET_PAR

        val_dict = {'C2H6_SRF': C2H6, 'ETH_SRF': ETH, 'PAR_SRF': PAR, 'OLET_SRF': OLET,
                    'TOL_SRF': TOL, 'XYL_SRF': XYL, 'CH3OH_SRF': CH3OH, 'ALD2_SRF': ALD2, 'AONE_SRF': AONE}

        res[key_str] = val_dict

    return res  


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


def write_result(text, time_rh, voc, aerosols, outfile):
    final_results = dict()
    for record in text:
        if(record in time_rh and record in voc and record in aerosols):
            resultant = text[record]
            resultant.update(time_rh[record])
            resultant.update(voc[record])
            resultant.update(aerosols[record])
            final_results[record] = resultant

    flipped_dict = dict()
    for key in final_results['2010-02-11 08:00:00'].keys():
        flipped_dict[key] = dict()
    for record in final_results:
        for metric in final_results[record]:
            flipped_dict[metric][record] = final_results[record][metric]

    df = pd.DataFrame(flipped_dict)
    df.to_csv(outfile)
    print('Data Normalization Complete!')


def create_numpy_array():
    array = np.array([0.0])
    array = np.delete(array, 0)
    return array


if __name__ == '__main__':
    main()
