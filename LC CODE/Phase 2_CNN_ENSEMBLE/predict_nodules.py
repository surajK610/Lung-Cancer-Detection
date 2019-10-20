import variables
import operators
import sys
import os
import glob
from typing import List, Tuple
from keras.optimizers import Adam, SGD
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, merge, Convolution3D, MaxPooling3D, UpSampling3D, LeakyReLU, BatchNormalization, Flatten, Dense, Dropout, ZeroPadding3D, AveragePooling3D, Activation
from keras.models import Model, load_model, model_from_json
from keras.metrics import binary_accuracy, binary_crossentropy, mean_squared_error, mean_absolute_error
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import math
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import nodule_detector
import random
import pandas
import ntpath
import cv2
import numpy



config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))



K.set_image_dim_ordering("tf")
SIZE = 48
MEAN_PIXEL_VALUE = variables.MEAN_PIXEL_VALUE_nod
NEGS_PER_POS = 20
P_TH = 0.6

PREDICT_STEP = 12
USE_DROPOUT = False


def prepare_image_for_net3D(img):
    img = img.astype(numpy.float32)
    img -= MEAN_PIXEL_VALUE
    img /= 255.
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
    return img


def filter_patient_nods_predictions(df_nod_predictions: pandas.DataFrame, patient_id, view_size, luna16=True):
    src_dir = variables.LUNA16_EXTRACTED_IMAGE_DIR if luna16 else variables.NDSB3_EXTRACTED_IMAGE_DIR
    patient_mask = operators.load_patient_imgs(patient_id, src_dir, "*_m.png")
    delete_indices = []
    for index, row in df_nod_predictions.iterrows():
        z_perc = row["coord_z"]
        y_perc = row["coord_y"]
        center_x = int(round(row["coord_x"] * patient_mask.shape[2]))
        center_y = int(round(y_perc * patient_mask.shape[1]))
        center_z = int(round(z_perc * patient_mask.shape[0]))

        mal_score = row["diameter_mm"]
        start_y = center_y - view_size / 2
        start_x = center_x - view_size / 2
        nod_in_mask = False
        for z_index in [-1, 0, 1]:
            img = patient_mask[z_index + center_z]
            start_x = int(start_x)
            start_y = int(start_y)
            view_size = int(view_size)
            img_roi = img[start_y:start_y+view_size, start_x:start_x + view_size]
            if img_roi.sum() > 255: 
                nod_in_mask = True

        if not nod_in_mask:
            print("nod not in mask: ", (center_x, center_y, center_z))
            if mal_score > 0:
                mal_score *= -1
            df_nod_predictions.loc[index, "diameter_mm"] = mal_score
        else:
            if center_z < 30:
                print("Z < 30: ", patient_id, " center z:", center_z, " y_perc: ",  y_perc)
                if mal_score > 0:
                    mal_score *= -1
                df_nod_predictions.loc[index, "diameter_mm"] = mal_score


            if (z_perc > 0.75 or z_perc < 0.25) and y_perc > 0.85:
                print("SUSPICIOUS FALSEPOSITIVE: ", patient_id, " center z:", center_z, " y_perc: ",  y_perc)

            if center_z < 50 and y_perc < 0.30:
                print("SUSPICIOUS FALSEPOSITIVE OUT OF RANGE: ", patient_id, " center z:", center_z, " y_perc: ",  y_perc)

    df_nod_predictions.drop(df_nod_predictions.index[delete_indices], inplace=True)
    return df_nod_predictions


def filter_nod_predictions(only_patient_id=None):
    src_dir = variables.NDSB3_nod_DETECTION_DIR
    for csv_index, csv_path in enumerate(glob.glob(src_dir + "*.csv")):
        file_name = ntpath.basename(csv_path)
        patient_id = file_name.replace(".csv", "")
        print(csv_index, ": ", patient_id)
        if only_patient_id is not None and patient_id != only_patient_id:
            continue
        df_nod_predictions = pandas.read_csv(csv_path)
        filter_patient_nods_predictions(df_nod_predictions, patient_id, SIZE)
        df_nod_predictions.to_csv(csv_path, index=False)


def make_negative_trn_data_based_on_predicted_luna_nods():
    src_dir = variables.LUNA_nod_DETECTION_DIR
    pos_labels_dir = variables.LUNA_nod_LABELS_DIR
    keep_dist = SIZE + SIZE / 2
    total_false_pos = 0
    for csv_index, csv_path in enumerate(glob.glob(src_dir + "*.csv")):
        file_name = ntpath.basename(csv_path)
        patient_id = file_name.replace(".csv", "")
        df_nod_predictions = pandas.read_csv(csv_path)
        pos_annos_manual = None
        manual_path = variables.MANUAL_ANNOTATIONS_LABELS_DIR + patient_id + ".csv"
        if os.path.exists(manual_path):
            pos_annos_manual = pandas.read_csv(manual_path)

        filter_patient_nods_predictions(df_nod_predictions, patient_id, SIZE, luna16=True)
        pos_labels = pandas.read_csv(pos_labels_dir + patient_id + "_annos_pos_lidc.csv")
        print(csv_index, ": ", patient_id, ", pos", len(pos_labels))
        patient_imgs = operators.load_patient_imgs(patient_id, variables.LUNA_16_trn_DIR2D2, "*_m.png")
        for nod_pred_index, nod_pred_row in df_nod_predictions.iterrows():
            if nod_pred_row["diameter_mm"] < 0:
                continue
            nx, ny, nz = operators.percentage_to_pixels(nod_pred_row["coord_x"], nod_pred_row["coord_y"], nod_pred_row["coord_z"], patient_imgs)
            diam_mm = nod_pred_row["diameter_mm"]
            for label_index, label_row in pos_labels.iterrows():
                px, py, pz = operators.percentage_to_pixels(label_row["coord_x"], label_row["coord_y"], label_row["coord_z"], patient_imgs)
                dist = math.sqrt(math.pow(nx - px, 2) + math.pow(ny - py, 2) + math.pow(nz- pz, 2))
                if dist < keep_dist:
                    if diam_mm >= 0:
                        diam_mm *= -1
                    df_nod_predictions.loc[nod_pred_index, "diameter_mm"] = diam_mm
                    break

            if pos_annos_manual is not None:
                for index, label_row in pos_annos_manual.iterrows():
                    px, py, pz = operators.percentage_to_pixels(label_row["x"], label_row["y"], label_row["z"], patient_imgs)
                    diameter = label_row["d"] * patient_imgs[0].shape[1]
                
                    dist = math.sqrt(math.pow(px - nx, 2) + math.pow(py - ny, 2) + math.pow(pz - nz, 2))
                    if dist < (diameter + 72):  
                        if diam_mm >= 0:
                            diam_mm *= -1
                        df_nod_predictions.loc[nod_pred_index, "diameter_mm"] = diam_mm
                        print("#Too close",  (nx, ny, nz))
                        break

        df_nod_predictions.to_csv(csv_path, index=False)
        df_nod_predictions = df_nod_predictions[df_nod_predictions["diameter_mm"] >= 0]
        df_nod_predictions.to_csv(pos_labels_dir + patient_id + "_candidates_falsepos.csv", index=False)
        total_false_pos += len(df_nod_predictions)
    print("Total false pos:", total_false_pos)


def predict_cubes(model_path, continue_job, only_patient_id=None, luna16=True, magnification=1, flip=False, trn_data=True, valid_no=-1, ext_name="", fold_count=2):
    if luna16:
        dst_dir = variables.LUNA_nod_DETECTION_DIR
    else:
        dst_dir = variables.NDSB3_nod_DETECTION_DIR + "2"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    valid_ext = ""
    flip_ext = ""
    if flip:
        flip_ext = "_flip"

    dst_dir += "predictions" + str(int(magnification * 10)) + valid_ext + flip_ext + "_" + ext_name + "/"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    sw = operators.Stopwatch.start_new()
    model = nodule_detector.get_net(input_shape=(SIZE, SIZE, SIZE, 1), load_weight_path=model_path)
    if luna16:
        labels_df = pandas.read_csv("/media/pikachu/Seagate Backup Plus Drive/LC nod Detection/resources/luna16_annotations/annotations.csv")
    if not luna16:
        if trn_data:
            labels_df = pandas.read_csv("resources/stage1_labels.csv")
            labels_df.set_index(["id"], inplace=True)
        else:
            labels_df = pandas.read_csv("resources/stage2_sample_submission.csv")
            labels_df.set_index(["id"], inplace=True)

    patient_ids = []
    for file_name in os.listdir(variables.LUNA16_EXTRACTED_IMAGE_DIR):
        if not os.path.isdir(variables.LUNA16_EXTRACTED_IMAGE_DIR + file_name):
            continue
        patient_ids.append(file_name)





    all_predictions_csv = []
    for patient_index, patient_id in enumerate(reversed(patient_ids)):
        if not luna16:
            if patient_id not in labels_df.index:
                continue
        if "metadata" in patient_id:
            continue
        if only_patient_id is not None and only_patient_id != patient_id:
            continue

        if valid_no is not None and trn_data:
            patient_fold = operators.get_patient_fold(patient_id)
            patient_fold %= fold_count
            if patient_fold != valid_no:
                continue

        print(patient_index, ": ", patient_id)
        csv_target_path = dst_dir + patient_id + ".csv"
        if continue_job and only_patient_id is None:
            if os.path.exists(csv_target_path):
                continue

        patient_img = operators.load_patient_imgs(patient_id, variables.LUNA16_EXTRACTED_IMAGE_DIR, "*_i.png", [])
        if magnification != 1:
            patient_img = operators.rescale_patient_imgs(patient_img, (1, 1, 1), magnification)

        patient_mask = operators.load_patient_imgs(patient_id, variables.LUNA16_EXTRACTED_IMAGE_DIR, "*_m.png", [])
        if magnification != 1:
            patient_mask = operators.rescale_patient_imgs(patient_mask, (1, 1, 1), magnification, is_mask_image=True)

        

        step = PREDICT_STEP
        CROP_SIZE = SIZE


        predict_volume_shape_list = [0, 0, 0]
        for dim in range(3):
            dim_indent = 0
            while dim_indent + CROP_SIZE < patient_img.shape[dim]:
                predict_volume_shape_list[dim] += 1
                dim_indent += step

        predict_volume_shape = (predict_volume_shape_list[0], predict_volume_shape_list[1], predict_volume_shape_list[2])
        predict_volume = numpy.zeros(shape=predict_volume_shape, dtype=float)
        print("Predict volume shape: ", predict_volume.shape)
        done_count = 0
        skipped_count = 0
        batch_size = 32
        batch_list = []
        batch_list_coords = []
        patient_predictions_csv = []
        cube_img = None
        annotation_index = 0

        for z in range(0, predict_volume_shape[0]):
            for y in range(0, predict_volume_shape[1]):
                for x in range(0, predict_volume_shape[2]):
                    cube_img = patient_img[z * step:z * step+CROP_SIZE, y * step:y * step + CROP_SIZE, x * step:x * step+CROP_SIZE]
                    cube_mask = patient_mask[z * step:z * step+CROP_SIZE, y * step:y * step + CROP_SIZE, x * step:x * step+CROP_SIZE]

                    if cube_mask.sum() < 2000:
                        skipped_count += 1
                    else:
                        if flip:
                            cube_img = cube_img[:, :, ::-1]

                        if CROP_SIZE != SIZE:
                            cube_img = operators.rescale_patient_imgs2(cube_img, (SIZE, SIZE, SIZE))
                            operators.save_cube_img("/media/pikachu/Seagate Backup Plus Drive/LC nod Detection/workdir", cube_img, 8, 4)
                            cube_mask = operators.rescale_patient_imgs2(cube_mask, (SIZE, SIZE, SIZE))

                        img_prep = prepare_image_for_net3D(cube_img)
                        batch_list.append(img_prep)
                        batch_list_coords.append((z, y, x))
                        if len(batch_list) % batch_size == 0:
                            batch_data = numpy.vstack(batch_list)
                            p = model.predict(batch_data, batch_size=batch_size)
                            for i in range(len(p[0])):
                                p_z = batch_list_coords[i][0]
                                p_y = batch_list_coords[i][1]
                                p_x = batch_list_coords[i][2]
                                nod_chance = p[0][i][0]
                                predict_volume[p_z, p_y, p_x] = nod_chance
                                if nod_chance > P_TH:
                                    p_z = p_z * step + CROP_SIZE / 2
                                    p_y = p_y * step + CROP_SIZE / 2
                                    p_x = p_x * step + CROP_SIZE / 2

                                    p_z_perc = round(p_z / patient_img.shape[0], 4)
                                    p_y_perc = round(p_y / patient_img.shape[1], 4)
                                    p_x_perc = round(p_x / patient_img.shape[2], 4)
                                    diameter_mm = round(p[1][i][0], 4)
                                    diameter_perc = round(2 * step / patient_img.shape[2], 4)
                                    diameter_perc = round(diameter_mm / patient_img.shape[2], 4)
                                    nod_chance = round(nod_chance, 4)
                                    patient_predictions_csv_line = [annotation_index, p_x_perc, p_y_perc, p_z_perc, diameter_perc, nod_chance, diameter_mm]
                                    patient_predictions_csv.append(patient_predictions_csv_line)
                                    all_predictions_csv.append([patient_id] + patient_predictions_csv_line)
                                    annotation_index += 1

                            batch_list = []
                            batch_list_coords = []
                    done_count += 1
                    if done_count % 10000 == 0:
                        print("Done: ", done_count, " skipped:", skipped_count)

        df = pandas.DataFrame(patient_predictions_csv, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "nod_chance", "diameter_mm"])
        print(df)
        if not df.empty:
            filter_patient_nods_predictions(df, patient_id, CROP_SIZE * magnification)
        df.to_csv(csv_target_path, index=False)

        print(predict_volume.mean())
        print("Done in : ", sw.get_elapsed_seconds(), " seconds")


def pipeline_main():
    CONTINUE_JOB = True
    only_patient_id = None  

    for magnification in [1, 1.5, 2]:  #
        predict_cubes("/media/pikachu/Seagate Backup Plus Drive/LC nod Detection/resources/stuff/trned_models/model_luna16_full__fs_best.hd5", CONTINUE_JOB, only_patient_id=only_patient_id, magnification=magnification, flip=False, trn_data=True, valid_no=None, ext_name="luna16_fs")
        predict_cubes("/media/pikachu/Seagate Backup Plus Drive/LC nod Detection/resources/stuff/trned_models/model_luna16_full__fs_best.hd5", CONTINUE_JOB, only_patient_id=only_patient_id, magnification=magnification, flip=False, trn_data=False, valid_no=None, ext_name="luna16_fs")

    for version in [2, 1]:
        for valid in [0, 1]:
            for magnification in [1, 2]:  #
                predict_cubes("/media/pikachu/Seagate Backup Plus Drive/LC nod Detection/resources/stuff/trned_models/model_luna_posnegndsb_v" + str(version) + "__fs_h" + str(valid) + "_end.hd5", CONTINUE_JOB, only_patient_id=only_patient_id, magnification=magnification, flip=False, trn_data=True, valid_no=valid, ext_name="luna_posnegndsb_v" + str(version), fold_count=2)
                if valid == 0:
                    predict_cubes("/media/pikachu/Seagate Backup Plus Drive/LC nod Detection/resources/stuff/trned_models/model_luna_posnegndsb_v" + str(version) + "__fs_h" + str(valid) + "_end.hd5", CONTINUE_JOB, only_patient_id=only_patient_id, magnification=magnification, flip=False, trn_data=False, valid_no=valid, ext_name="luna_posnegndsb_v" + str(version), fold_count=2)


if __name__ == "__main__":
    pipeline_main()
  