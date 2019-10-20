import variables
import operators
import sys
import os
import glob
import random
import pandas
import ntpath
import cv2
import numpy
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
import shutil
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session





config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))


SIZE = 48
MEAN_PIXEL_VALUE = variables.MEAN_PIXEL_VALUE_nod
POS_WEIGHT = 2
NEGS_PER_POS = 20
P_TH = 0.6
l_r = 0.001

K.set_image_dim_ordering("tf")


def prepare_image_for_net3D(img):
    img = img.astype(numpy.float32)
    img -= MEAN_PIXEL_VALUE
    img /= 255.
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
    return img


def get_trn_valid_files(fold_count, trn_percentage=80, logreg=True, ndsb3_valid=0, manual_labels=True, full_luna_set=False):
    print("Get trn/valid files.")
    pos_samples = glob.glob(variables.BASE_DIR_SSD + "generated_trndata/luna16_trn_cubes_lidc/*.png")
    pos_samples_manual = glob.glob(variables.BASE_DIR_SSD + "generated_trndata/luna16_trn_cubes_manual/*_pos.png")
    pos_samples += pos_samples_manual

    random.shuffle(pos_samples)
    trn_pos_count = int((len(pos_samples) * trn_percentage) / 100)
    pos_samples_trn = pos_samples[:trn_pos_count]
    pos_samples_valid = pos_samples[trn_pos_count:]
    if full_luna_set:
        pos_samples_trn += pos_samples_valid
        if manual_labels:
            pos_samples_valid = []


    ndsb3_list = glob.glob(variables.BASE_DIR_SSD + "generated_trndata/ndsb3_trn_cubes_manual/*.png")
    print("Ndsb3 samples: ", len(ndsb3_list))

    pos_samples_ndsb3_fold = []
    pos_samples_ndsb3_valid = []
    ndsb3_pos = 0
    ndsb3_neg = 0
    ndsb3_pos_valid = 0
    ndsb3_neg_valid = 0
    if manual_labels:
        for file_path in ndsb3_list:
            file_name = ntpath.basename(file_path)

            parts = file_name.split("_")
            if int(parts[4]) == 0 and parts[3] != "neg": 
                continue

            if fold_count == 3:
                if parts[3] == "neg": 
                    continue


            patient_id = parts[1]
            patient_fold = operators.get_patient_fold(patient_id) % fold_count
            if patient_fold == ndsb3_valid:
                pos_samples_ndsb3_valid.append(file_path)
                if parts[3] == "neg":
                    ndsb3_neg_valid += 1
                else:
                    ndsb3_pos_valid += 1
            else:
                pos_samples_ndsb3_fold.append(file_path)
                print("In fold: ", patient_id)
                if parts[3] == "neg":
                    ndsb3_neg += 1
                else:
                    ndsb3_pos += 1

    print(ndsb3_pos, " ndsb3 pos labels trn")
    print(ndsb3_neg, " ndsb3 neg labels trn")
    print(ndsb3_pos_valid, " ndsb3 pos labels valid")
    print(ndsb3_neg_valid, " ndsb3 neg labels valid")


    if manual_labels:
        for times_ndsb3 in range(4):  
            pos_samples_trn += pos_samples_ndsb3_fold
            pos_samples_valid += pos_samples_ndsb3_valid

    neg_samples_edge = glob.glob(variables.BASE_DIR_SSD + "generated_traindata/luna16_train_cubes_auto/*_edge.png")
    neg_samples_luna = glob.glob(variables.BASE_DIR_SSD + "generated_traindata/luna16_train_cubes_auto/*_luna.png")

    neg_samples = neg_samples_edge + neg_samples_luna
    random.shuffle(neg_samples)

    trn_neg_count = int((len(neg_samples) * trn_percentage) / 100)

    neg_samples_falsepos = []
    for file_path in glob.glob(variables.BASE_DIR_SSD + "generated_traindata/luna16_train_cubes_auto/*_falsepos.png"):
        neg_samples_falsepos.append(file_path)

    neg_samples_trn = neg_samples[:trn_neg_count]
    neg_samples_trn += neg_samples_falsepos + neg_samples_falsepos + neg_samples_falsepos
    neg_samples_valid = neg_samples[trn_neg_count:]
    if full_luna_set:
        neg_samples_trn += neg_samples_valid

    trn_res = []
    valid_res = []
    sets = [(trn_res, pos_samples_trn, neg_samples_trn), (valid_res, pos_samples_valid, neg_samples_valid)]
    for set_item in sets:
        pos_idx = 0
        negs_per_pos = NEGS_PER_POS
        res = set_item[0]
        neg_samples = set_item[2]
        pos_samples = set_item[1]
        ndsb3_pos = 0
        ndsb3_neg = 0
        for index, neg_sample_path in enumerate(neg_samples):
            res.append((neg_sample_path, 0, 0))
            if index % negs_per_pos == 0:
                pos_sample_path = pos_samples[pos_idx]
                file_name = ntpath.basename(pos_sample_path)
                parts = file_name.split("_")
                if parts[0].startswith("ndsb3manual"):
                    if parts[3] == "pos":
                        class_label = 1 
                        cancer_label = int(parts[4])
                        ndsb3_pos += 1
                    else:
                        class_label = 0
                        size_label = 0
                        ndsb3_neg += 1
                else:
                    class_label = int(parts[-2])
                    size_label = int(parts[-3])


                res.append((pos_sample_path, class_label, size_label))
                pos_idx += 1
                pos_idx %= len(pos_samples)


    print("trn count: ", len(trn_res), ", valid count: ", len(valid_res))
    return trn_res, valid_res


##MUCH OF THe DATA_GENERATOR FUNCTION IS BORROWED AND MODIFIED BY JULIAN DE WIT
def data_generator(batch_size, record_list, trn_set):
    batch_idx = 0
    means = []
    random_state = numpy.random.RandomState(1301)
    while True:
        img_list = []
        class_list = []
        size_list = []
        if trn_set:
            random.shuffle(record_list)
        CROP_SIZE = SIZE
        for record_idx, record_item in enumerate(record_list):
            class_label = record_item[1]
            size_label = record_item[2]
            if class_label == 0:
                cube_image = operators.load_cube_img(record_item[0], 6, 8, 48)
            
                wiggle = 48 - CROP_SIZE - 1
                indent_x = 0
                indent_y = 0
                indent_z = 0
                if wiggle > 0:
                    indent_x = random.randint(0, wiggle)
                    indent_y = random.randint(0, wiggle)
                    indent_z = random.randint(0, wiggle)
                cube_image = cube_image[indent_z:indent_z + CROP_SIZE, indent_y:indent_y + CROP_SIZE, indent_x:indent_x + CROP_SIZE]

                if trn_set:
                    if random.randint(0, 100) > 50:
                        cube_image = numpy.fliplr(cube_image)
                    if random.randint(0, 100) > 50:
                        cube_image = numpy.flipud(cube_image)
                    if random.randint(0, 100) > 50:
                        cube_image = cube_image[:, :, ::-1]
                    if random.randint(0, 100) > 50:
                        cube_image = cube_image[:, ::-1, :]

                if CROP_SIZE != SIZE:
                    cube_image = operators.rescale_patient_imgs2(cube_image, (SIZE, SIZE, SIZE))
                assert cube_image.shape == (SIZE, SIZE, SIZE)
            else:
                cube_image = operators.load_cube_img(record_item[0], 8, 8, 64)

                if trn_set:
                    pass

                current_SIZE = cube_image.shape[0]
                indent_x = (current_SIZE - CROP_SIZE) / 2
                indent_y = (current_SIZE - CROP_SIZE) / 2
                indent_z = (current_SIZE - CROP_SIZE) / 2
                wiggle_indent = 0
                wiggle = current_SIZE - CROP_SIZE - 1
                if wiggle > (CROP_SIZE / 2):
                    wiggle_indent = CROP_SIZE / 4
                    wiggle = current_SIZE - CROP_SIZE - CROP_SIZE / 2 - 1
                if trn_set:
                    indent_x = wiggle_indent + random.randint(0, wiggle)
                    indent_y = wiggle_indent + random.randint(0, wiggle)
                    indent_z = wiggle_indent + random.randint(0, wiggle)

                indent_x = int(indent_x)
                indent_y = int(indent_y)
                indent_z = int(indent_z)
                cube_image = cube_image[indent_z:indent_z + CROP_SIZE, indent_y:indent_y + CROP_SIZE, indent_x:indent_x + CROP_SIZE]
                if CROP_SIZE != SIZE:
                    cube_image = operators.rescale_patient_imgs2(cube_image, (SIZE, SIZE, SIZE))
                assert cube_image.shape == (SIZE, SIZE, SIZE)

                if trn_set:
                    if random.randint(0, 100) > 50:
                        cube_image = numpy.fliplr(cube_image)
                    if random.randint(0, 100) > 50:
                        cube_image = numpy.flipud(cube_image)
                    if random.randint(0, 100) > 50:
                        cube_image = cube_image[:, :, ::-1]
                    if random.randint(0, 100) > 50:
                        cube_image = cube_image[:, ::-1, :]


            means.append(cube_image.mean())
            img3d = prepare_image_for_net3D(cube_image)
            if trn_set:
                if len(means) % 1000000 == 0:
                    print("Mean: ", sum(means) / len(means))
            img_list.append(img3d)
            class_list.append(class_label)
            size_list.append(size_label)

            batch_idx += 1
            if batch_idx >= batch_size:
                x = numpy.vstack(img_list)
                y_class = numpy.vstack(class_list)
                y_size = numpy.vstack(size_list)
                yield x, {"out_class": y_class, "out_malignancy": y_size}
                img_list = []
                class_list = []
                size_list = []
                batch_idx = 0


def FCN_Vgg16_32s(input_shape=(SIZE, SIZE, SIZE, 1), weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    #classifying layer
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)

    x = BilinearUpSampling2D(size=(32, 32))(x)

    model = Model(img_input, x)

    return model


def build_cnn_morph(input_shape=(SIZE, SIZE, SIZE, 1)):

    inputs = Input(input_shape)
    
    a = Convolution3D(8,3,3,3,border_mode='same',W_regularizer=l2(1e-4))(inputs)
    a = BatchNormalization(axis=1)(a)
    a1 = AveragePooling3D()(a)
    a = merge([a, a1],mode='concat', concat_axis=1)
    
    b = Convolution3D(24,3,3,3,border_mode='same',W_regularizer=l2(1e-4))(a)
    b = BatchNormalization(axis=1)(b)
    b1 = AveragePooling3D()(a1)
    b = merge([b, b1],mode='concat', concat_axis=1)

    c = Convolution3D(64,3,3,3,border_mode='same',W_regularizer=l2(1e-4))(b)
    c = BatchNormalization(axis=1)(c)
    c1 = AveragePooling3D()(b1)
    c = merge([c, c1],mode='concat', concat_axis=1)

    d = Convolution3D(72,3,3,3,border_mode='same',W_regularizer=l2(1e-4))(c)
    d = BatchNormalization(axis=1)(c)
    d1 = AveragePooling3D()(c1)
    d = merge([d, d1],mode='concat', concat_axis=1)
    
    e = Convolution3D(72,3,3,3,border_mode='same',W_regularizer=l2(1e-4))(d)
    e = BatchNormalization(axis=1)(e)
    
    pool = GlobalMaxPooling3D()(e)
    pool_norm = BatchNormalization()(pool)

    diam= dense_branch(pool_norm,name='o_d',outsize=1,activation='relu')
    cad_falsepositive = dense_branch(pool_norm, name='o_fp',outsize=3,activation='softmax')
    margin = dense_branch(pool_norm,name='o_marg',outsize=1,activation='sigmoid')
    lob = dense_branch(pool_norm,name='o_lob',outsize=1,activation='sigmoid')
    pic = dense_branch(pool_norm,name='o_spic',outsize=1,activation='sigmoid')
    malig = dense_branch(pool_norm,name='o_mal',outsize=1,activation='sigmoid')
    
    model = Model(input=inputs,output=[diam, lob, spic, malig, cad_falsepositive])
    
    if input_shape[1] == 32:
        lr_start = .005
    elif input_shape[1] == 64:
        lr_start = .001
    elif input_shape[1] == 128:
        lr_start = 1e-4
    elif input_shape[1] == 96:
        lr_start = 5e-4
    
        
    opt = Nadam(lr_start,clipvalue=1.0)
    model.compile(optimizer=opt,loss={'o_d':'mse', 'o_lob':'binary_crossentropy', 'o_spic':'binary_crossentropy', 
                                        'o_mal':'binary_crossentropy', 'o_fp':'categorical_crossentropy'},
                                loss_weights={'o_d':1.0, 'o_lob':5.0, 'o_spic':5.0, 'o_mal':5.0, 'o_fp':5.0})

    return model


#MODEL STURCTURE FROM JULIAN DE WIT
def get_vgg_net(input_shape=(SIZE, SIZE, SIZE, 1), load_weight_path=None, features=False, mal=False):
    inputs = Input(shape=input_shape, name="input_1")
    x = inputs
    x = AveragePooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1), border_mode="same")(x)
    x = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same', name='conv1', subsample=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), border_mode='valid', name='pool1')(x)

    x = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same', name='conv2', subsample=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool2')(x)
    x = Dropout(p=0.2)(x)

    x = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv3a', subsample=(1, 1, 1))(x)
    x = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv3b', subsample=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool3')(x)
    x = Dropout(p=0.3)(x)

    x = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv4a', subsample=(1, 1, 1))(x)
    x = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv4b', subsample=(1, 1, 1),)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool4')(x)
    x = Dropout(p=0.4)(x)

    l64 = Convolution3D(64, 2, 2, 2, activation="relu", name="l64")(x)
    out_class = Convolution3D(1, 1, 1, 1, activation="sigmoid", name="out_class_last")(l64)
    out_class = Flatten(name="out_class")(out_class)

    out_malignancy = Convolution3D(1, 1, 1, 1, activation=None, name="out_malignancy_last")(l64)
    out_malignancy = Flatten(name="out_malignancy")(out_malignancy)

    model = Model(input=inputs, output=[out_class, out_malignancy])
    if load_weight_path is not None:
        model.load_weights(load_weight_path, by_name=False)
    model.compile(optimizer=SGD(lr=l_r, momentum=0.9, nesterov=True), loss={"out_class": "binary_crossentropy", "out_malignancy": mean_absolute_error}, metrics={"out_class": [binary_accuracy, binary_crossentropy], "out_malignancy": mean_absolute_error})

    if features:
        model = Model(input=inputs, output=[last64])
    model.summary(line_length=140)

    return model


def step_decay(epoch):
    res = 0.001
    if epoch > 5:
        res = 0.0001
    print("learnrate: ", res, " epoch: ", epoch)
    return res

def trn_models(model_name, fold_count, trn_full_set=False, load_weights_path=None, ndsb3_valid=0, manual_labels=True):        
    batch_size = 16

    df = pd.read_csv("workdir/annotations_e.csv")
    Ydiam = df['diameter_mm'].values.astype('float32')
    Ycalc = np.zeros((df.shape[0],7)).astype('float32')
    Yspher = np.zeros((df.shape[0],4)).astype('float32')
    Ytext = np.zeros((df.shape[0],4)).astype('float32')
        
    df['calcification'] = df['calcification'].apply(lambda x: str_to_arr(x,6))
    df['sphericity'] = df['sphericity'].apply(lambda x: str_to_arr(x,3))
    df['texture'] = df['texture'].apply(lambda x: str_to_arr(x,3))
    
    trn_files, valid_files = get_trn_valid_files(trn_percentage=80, ndsb3_valid=False, manual_labels=manual_labels, full_luna_set=trn_full_set, fold_count=fold_count)

    checkpoint = ModelCheckpoint("workdir/modeln_" + model_name + "_" + valid_txt + "_e" + "{epoch:02d}-{val_loss:.4f}.hd5", monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
    checkpoint_name = ModelCheckpoint("workdir/modeln_" + model_name + "_" + valid_txt + "_best.hd5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
    model = build_cnn_morph()

    nb_epoch=15

    for epoch in range(nb_epoch):
        if split:
            model.fit_generator(train_generatorn,samples_per_epoch=samples_per_epoch*batch_size, nb_epoch=epoch+1,callbacks=[checkpoint, checkpoint_name],
                                validation_data=valid_generatorn, nb_val_samples=samples_per_epoch*batch_size/2,initial_epoch=epoch)
        else:
            model.fit_generator(train_generatorn,samples_per_epoch=samples_per_epoch*batch_size, nb_epoch=epoch+1,callbacks=[checkpoint, checkpoint_name],
                                initial_epoch=epoch)
    
    model.save("workdir/models_" + model_name + "_" + valid_txt + "_end.hd5")


def trn_modelf(model_name, fold_count, trn_full_set=False, load_weights_path=None, ndsb3_valid=0, manual_labels=True):        
    batch_size = 16
    trn_files, valid_files = get_trn_valid_files(trn_percentage=80, ndsb3_valid=False, manual_labels=manual_labels, full_luna_set=trn_full_set, fold_count=fold_count)


    trn_gen = data_generator(batch_size, trn_files, True)
    valid_gen = data_generator(batch_size, valid_files, False)
    for i in range(0, 10):
        tmp = next(valid_gen)
        cube_img = tmp[0][0].reshape(SIZE, SIZE, SIZE, 1)
        cube_img = cube_img[:, :, :, 0]
        cube_img *= 255.
        cube_img += MEAN_PIXEL_VALUE
                
    checkpoint = ModelCheckpoint("workdir/modeln_" + model_name + "_" + valid_txt + "_e" + "{epoch:02d}-{val_loss:.4f}.hd5", monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
    checkpoint_name = ModelCheckpoint("workdir/modeln_" + model_name + "_" + valid_txt + "_best.hd5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
    
    nb_epoch=15

    for epoch in range(nb_epoch):
        if split:
            model.fit_generator(train_generatorn,samples_per_epoch=samples_per_epoch*batch_size, nb_epoch=epoch+1,callbacks=[checkpoint, checkpoint_name],
                                validation_data=valid_generatorn, nb_val_samples=samples_per_epoch*batch_size/2,initial_epoch=epoch)
        else:
            model.fit_generator(train_generatorn,samples_per_epoch=samples_per_epoch*batch_size, nb_epoch=epoch+1,callbacks=[checkpoint, checkpoint_name],
                                initial_epoch=epoch)
    
    model.save("workdir/modelf_" + model_name + "_" + valid_txt + "_end.hd5")



def trn(model_name, fold_count, trn_full_set=False, load_weights_path=None, ndsb3_valid=0, manual_labels=True):
    batch_size = 16
    trn_files, valid_files = get_trn_valid_files(trn_percentage=80, ndsb3_valid=ndsb3_valid, manual_labels=manual_labels, full_luna_set=trn_full_set, fold_count=fold_count)


    trn_gen = data_generator(batch_size, trn_files, True)
    valid_gen = data_generator(batch_size, valid_files, False)
    for i in range(0, 10):
        tmp = next(valid_gen)
        cube_img = tmp[0][0].reshape(SIZE, SIZE, SIZE, 1)
        cube_img = cube_img[:, :, :, 0]
        cube_img *= 255.
        cube_img += MEAN_PIXEL_VALUE
  

    model = get_vgg_net(load_weight_path=load_weights_path)
    valid_txt = "_h" + str(ndsb3_valid) if manual_labels else ""
    if trn_full_set:
        valid_txt = "_fs" + valid_txt
    checkpoint = ModelCheckpoint("workdir/model_" + model_name + "_" + valid_txt + "_e" + "{epoch:02d}-{val_loss:.4f}.hd5", monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
    checkpoint_name = ModelCheckpoint("workdir/model_" + model_name + "_" + valid_txt + "_best.hd5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
    model.fit_generator(trn_gen, len(trn_files) / 1, 12, validation_data=valid_gen, nb_val_samples=len(valid_files) / 1, callbacks=[checkpoint, checkpoint_name])
    model.save("workdir/model_" + model_name + "_" + valid_txt + "_end.hd5")


def pipeline_main():
    if not os.path.exists("models/"):
        os.mkdir("models")
    trn_modelf(model_name="luna16_fully", trn_full_set=True, load_weights_path=None, ndsb3_valid=0, manual_labels=False, fold_count=-1)
    trn_models(model_name="luna16_full_morph", trn_full_set=True, load_weights_path=None, ndsb3_valid=0, manual_labels=False, fold_count=-1)

    trn(trn_full_set=True, load_weights_path=None, model_name="luna16_full", fold_count=-1, manual_labels=False)
    trn(trn_full_set=True, load_weights_path=None, ndsb3_valid=0, manual_labels=True, model_name="lunapnANDndsb1", fold_count=2)
    trn(trn_full_set=True, load_weights_path=None, ndsb3_valid=1, manual_labels=True, model_name="lunapnANDndsb1", fold_count=2)
    trn(trn_full_set=True, load_weights_path=None, ndsb3_valid=0, manual_labels=True, model_name="luna_posnegndsb_v2", fold_count=2)
    trn(trn_full_set=True, load_weights_path=None, ndsb3_valid=1, manual_labels=True, model_name="luna_posnegndsb_v2", fold_count=2)


if __name__ == "__main__":
    pipeline_main()

