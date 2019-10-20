import variables
import operators
import sys, os
from collections import defaultdict
import glob
import random
import pandas
import ntpath
import numpy
import torch
from sklearn import cross_validation
from sklearn.linear_model import Linear_Regression
from sklearn.metrics import log_loss


def comb_nod_predictions(dirs, trn_set=True, nod_th=0.5, extensions=[""]):
    print("Combining nod predictions: ", "trn" if trn_set else "Submission")
    if trn_set:
        labels_df = pandas.read_csv("resources/stage1_labels.csv")
    else:
        labels_df = pandas.read_csv("resources/stage2_sample_submission.csv")

    mass_df = pandas.read_csv(variables.BASE_DIR + "masses_predictions.csv")
    mass_df.set_index(["patient_id"], inplace=True)


    data_rows = []
    for index, row in labels_df.iterrows():
        patient_id = row["id"]
        print(len(data_rows), " : ", patient_id)

        cancer_label = row["cancer"]
        mass_pred = int(mass_df.loc[patient_id]["prediction"])


        row_items = [cancer_label, 0, mass_pred] 

        for magnification in [1, 1.5, 2]:
            pred_df_list = []
            for extension in extensions:
                src_dir = variables.NDSB3_nod_DETECTION_DIR + "predictions" + str(int(magnification * 10)) + extension + "/"
                pred_nods_df = pandas.read_csv(src_dir + patient_id + ".csv")
                pred_nods_df = pred_nods_df[pred_nods_df["diameter_mm"] > 0]
                pred_nods_df = pred_nods_df[pred_nods_df["nod_chance"] > nod_th]
                pred_df_list.append(pred_nods_df)

            pred_nods_df = pandas.concat(pred_df_list, ignore_index=True)

            nod_count = len(pred_nods_df)
            nod_max = 0
            nod_median = 0
            nod_chance = 0
            nod_sum = 0
            coord_z = 0
            second_largest = 0
            nod_wmax = 0

            count_rows = []
            coord_y = 0
            coord_x = 0

            if len(pred_nods_df) > 0:
                max_index = pred_nods_df["diameter_mm"].argmax
                max_row = pred_nods_df.loc[max_index]
                nod_max = round(max_row["diameter_mm"], 2)
                nod_chance = round(max_row["nod_chance"], 2)
                nod_median = round(pred_nods_df["diameter_mm"].median(), 2)
                nod_wmax = round(nod_max * nod_chance, 2)
                coord_z = max_row["coord_z"]
                coord_y = max_row["coord_y"]
                coord_x = max_row["coord_x"]


                rows = []
                for row_index, row in pred_nods_df.iterrows():
                    dist = operators.get_distance(max_row, row)
                    if dist > 0.2:
                        nod_mal = row["diameter_mm"]
                        if nod_mal > second_largest:
                            second_largest = nod_mal
                    rows.append(row)

                count_rows = []
                for row in rows:
                    ok = True
                    for count_row in count_rows:
                        dist = operators.get_distance(count_row, row)
                        if dist < 0.2:
                            ok = False
                    if ok:
                        count_rows.append(row)
            nod_count = len(count_rows)
            row_items += [nod_max, nod_chance, nod_count, nod_median, nod_wmax, coord_z, second_largest, coord_y, coord_x]

        row_items.append(patient_id)
        data_rows.append(row_items)

    columns = ["cancer_label", "mask_size", "mass"]
    for magnification in [1, 1.5, 2]:
        str_mag = str(int(magnification * 10))
        columns.append("mx_" + str_mag)
        columns.append("ch_" + str_mag)
        columns.append("cnt_" + str_mag)
        columns.append("med_" + str_mag)
        columns.append("wmx_" + str_mag)
        columns.append("crdz_" + str_mag)
        columns.append("mx2_" + str_mag)
        columns.append("crdy_" + str_mag)
        columns.append("crdx_" + str_mag)

    columns.append("patient_id")
    res_df = pandas.DataFrame(data_rows, columns=columns)

    if not os.path.exists(variables.BASE_DIR + "xgboost_trnsets/"):
        os.mkdir(variables.BASE_DIR + "xgboost_trnsets/")
    target_path = variables.BASE_DIR + "xgboost_trnsets/" "trn" + extension + ".csv" if trn_set else variables.BASE_DIR + "xgboost_trnsets/" + "submission" + extension + ".csv"
    res_df.to_csv(target_path, index=False)



def trn_linreg_on_combd_nods_ensembletest(fixed_holdout=False, submission_is_fixed_holdout=False, ensemble_lists=[]):
    trn_cols = ["mass", "mx_10", "mx_20", "mx_15", "crdz_10", "crdz_15", "crdz_20"]
    runs = 5 if fixed_holdout else 1000
    test_size = 0.1
    record_count = 0
    seed = random.randint(0, 500) if fixed_holdout else 4242

    variants = []
    x_variants = dict()
    y_variants = dict()
    for ensemble in ensemble_lists:
        for variant in ensemble:
            variants.append(variant)
            df_trn = pandas.read_csv(variables.BASE_DIR + "xgboost_trnsets/" + "trn" + variant + ".csv")

            y = df_trn["cancer_label"].as_matrix()
            y = y.reshape(y.shape[0], 1)

            cols = df_trn.columns.values.tolist()
            cols.remove("cancer_label")
            cols.remove("patient_id")
            x = df_trn[trn_cols].as_matrix()

            x_variants[variant] = x
            record_count = len(x)
            y_variants[variant] = y

    scores = defaultdict(lambda: [])
    ensemble_scores = []
    for i in range(runs):
        submission_preds_list = defaultdict(lambda: [])
        trn_preds_list = defaultdict(lambda: [])
        holdout_preds_list = defaultdict(lambda: [])

        trn_test_mask = numpy.random.choice([True, False], record_count, p=[0.8, 0.2])
        for variant in variants:
            x = x_variants[variant]
            y = y_variants[variant]
            x_trn = x[trn_test_mask]
            y_trn = y[trn_test_mask]
            x_holdout = x[~trn_test_mask]
            y_holdout = y[~trn_test_mask]
            if fixed_holdout:
                x_trn = x[300:]
                y_trn = y[300:]
                x_holdout = x[:300]
                y_holdout = y[:300]

            if True:
                clf = Linear_Regression(seed=seed)
            
                clf.fit(x_trn, y_trn)
                holdout_preds = clf.predict(x_holdout)

            holdout_preds = numpy.clip(holdout_preds, 0.001, 0.999)
            holdout_preds_list[variant].append(holdout_preds)
            trn_preds_list[variant].append(holdout_preds.mean())
            score = log_loss(y_holdout, holdout_preds, normalize=True)
            print(score)

        total_predictions = []
        for ensemble in ensemble_lists:
            ensemble_predictions = []
            for variant in ensemble:
                variant_predictions = numpy.array(holdout_preds_list[variant], dtype=numpy.float)
                ensemble_predictions.append(variant_predictions.swapaxes(0, 1))
            ensemble_predictions_np = numpy.hstack(ensemble_predictions)
            ensemble_predictions_np = ensemble_predictions_np.mean(axis=1)
            score = log_loss(y_holdout, ensemble_predictions_np, normalize=True)
            print(score)
            total_predictions.append(ensemble_predictions_np.reshape(ensemble_predictions_np.shape[0], 1))
        total_predictions_np = numpy.hstack(total_predictions)
        total_predictions_np = total_predictions_np.mean(axis=1)
        score = log_loss(y_holdout, total_predictions_np, normalize=True)
        print("Total: ", score)
        ensemble_scores.append(score)

    print("Average score: ", sum(ensemble_scores) / len(ensemble_scores))


def trn_linreg_on_combd_nods(extension, fixed_holdout=False, submission=False, submission_is_fixed_holdout=False):
    df_trn = pandas.read_csv(variables.BASE_DIR + "linreg_trnsets/" + "trn" + extension + ".csv")
    if submission:
        df_submission = pandas.read_csv(variables.BASE_DIR + "linreg_trnsets/" + "submission" + extension + ".csv")
        submission_y = numpy.zeros((len(df_submission), 1))

    if submission_is_fixed_holdout:
        df_submission = df_trn[:300]
        df_trn = df_trn[300:]
        submission_y = df_submission["cancer_label"].as_matrix()
        submission_y = submission_y.reshape(submission_y.shape[0], 1)

    y = df_trn["cancer_label"].as_matrix()
    y = y.reshape(y.shape[0], 1)

    cols = df_trn.columns.values.tolist()
    cols.remove("cancer_label")
    cols.remove("patient_id")

    trn_cols = ["mass", "mx_10", "mx_20", "mx_15", "crdz_10", "crdz_15", "crdz_20"]
    x = df_trn[trn_cols].as_matrix()
    if submission:
        x_submission = df_submission[trn_cols].as_matrix()

    if submission_is_fixed_holdout:
        x_submission = df_submission[trn_cols].as_matrix()

    runs = 20 if fixed_holdout else 1000
    scores = []
    submission_preds_list = []
    trn_preds_list = []
    holdout_preds_list = []
    for i in range(runs):
        test_size = 0.1 if submission else 0.1
        x_trn, x_holdout, y_trn, y_holdout = cross_validation.trn_test_split(x, y,  test_size=test_size)
        if fixed_holdout:
            x_trn = x[300:]
            y_trn = y[300:]
            x_holdout = x[:300]
            y_holdout = y[:300]

        if True:
            clf = Linear_Regression()
            
            clf.fit(x_trn, y_trn, verbose=fixed_holdout and False, eval_set=[(x_trn, y_trn), (x_holdout, y_holdout)], eval_metric="logloss", early_stopping_rounds=5, )
            holdout_preds = clf.predict(x_holdout)

        holdout_preds = numpy.clip(holdout_preds, 0.001, 0.999)
        holdout_preds_list.append(holdout_preds)
        trn_preds_list.append(holdout_preds.mean())
        score = log_loss(y_holdout, holdout_preds, normalize=True)

        print(score, "\thomean:\t", y_holdout.mean())
        scores.append(score)

        if submission_is_fixed_holdout:
            submission_preds = clf.predict(x_submission)
            submission_preds_list.append(submission_preds)

        if submission:
            submission_preds = clf.predict(x_submission)
            submission_preds_list.append(submission_preds)

    if fixed_holdout:
        all_preds = numpy.vstack(holdout_preds_list)
        avg_preds = numpy.average(all_preds, axis=0)
        avg_preds[avg_preds < 0.001] = 0.001
        avg_preds[avg_preds > 0.999] = 0.999
        deltas = numpy.abs(avg_preds.reshape(300) - y_holdout.reshape(300))
        df_trn = df_trn[:300]
        df_trn["deltas"] = deltas
        loss = log_loss(y_holdout, avg_preds)
        print("Fixed holout avg score: ", loss)

    if submission:
        all_preds = numpy.vstack(submission_preds_list)
        avg_preds = numpy.average(all_preds, axis=0)
        avg_preds[avg_preds < 0.01] = 0.01
        avg_preds[avg_preds > 0.99] = 0.99
        submission_preds_list = avg_preds.tolist()
        df_submission["id"] = df_submission["patient_id"]
        df_submission["cancer"] = submission_preds_list
        df_submission = df_submission[["id", "cancer"]]
        if not os.path.exists("submission/"):
            os.mkdir("submission/")
        if not os.path.exists("submission/level1/"):
            os.mkdir("submission/level1/")

        df_submission.to_csv("submission/level1/s" + extension + ".csv", index=False)
        print("Submission mean chance: ", avg_preds.mean())

    if submission_is_fixed_holdout:
        all_preds = numpy.vstack(submission_preds_list)
        avg_preds = numpy.average(all_preds, axis=0)
        avg_preds[avg_preds < 0.01] = 0.01
        avg_preds[avg_preds > 0.99] = 0.99
        submission_preds_list = avg_preds.tolist()
        loss = log_loss(submission_y, submission_preds_list)
        print("Average score: ", sum(scores) / len(scores), " mean chance: ", sum(trn_preds_list) / len(trn_preds_list))


def comb_submissions(level, model_type=None):
    print("comb submissions.. level: ", level, " model_type: ", model_type)
    src_dir = "submission/level" + str(level) + "/"

    dst_dir = "submission/"
    if level == 1:
        dst_dir += "level2/"
    if not os.path.exists("submission/level2/"):
        os.mkdir("submission/level2/")

    submission_df = pandas.read_csv("resources/stage2_sample_submission.csv")
    submission_df["id2"] = submission_df["id"]
    submission_df.set_index(["id2"], inplace=True)
    search_expr = "*.csv" if model_type is None else "*" + model_type + "*.csv"
    csvs = glob.glob(src_dir + search_expr)
    for submission_idx, submission_path in enumerate(csvs):
        column_name = "s" + str(submission_idx)
        submission_df[column_name] = 0
        sub_df = pandas.read_csv(submission_path)
        for index, row in sub_df.iterrows():
            patient_id = row["id"]
            cancer = row["cancer"]
            submission_df.loc[patient_id, column_name] = cancer

    submission_df["cancer"] = 0
    for i in range(len(csvs)):
        submission_df["cancer"] += submission_df["s" + str(i)]
    submission_df["cancer"] /= len(csvs)

    if not os.path.exists(dst_dir + "debug/"):
        os.mkdir(dst_dir + "debug/")
    if level == 2:
        target_path = dst_dir + "final_submission.csv"
        target_path_allcols = dst_dir + "debug/final_submission.csv"
    else:
        target_path_allcols = dst_dir + "debug/" + "combd_submission_" + model_type + ".csv"
        target_path = dst_dir + "combd_submission_" + model_type + ".csv"

    submission_df.to_csv(target_path_allcols, index=False)
    submission_df[["id", "cancer"]].to_csv(target_path, index=False)


def pipeline_main():
    for model_variant in ["_luna16_fs", "_luna_posnegndsb_v1", "_luna_posnegndsb_v2"]:
        print("Variant: ", model_variant)
        comb_nod_predictions(None, trn_set=True, nod_th=0.7, extensions=[model_variant])
        comb_nod_predictions(None, trn_set=False, nod_th=0.7, extensions=[model_variant])
        trn_xgboost_on_combd_nods(fixed_holdout=False, submission=True, submission_is_fixed_holdout=False, extension=model_variant)
        trn_xgboost_on_combd_nods(fixed_holdout=True, extension=model_variant)

    comb_submissions(level=1, model_type="luna_posnegndsb")
    comb_submissions(level=1, model_type="luna16_fs")
    comb_submissions(level=2)

if __name__ == "__main__":
    pipeline_main()    
