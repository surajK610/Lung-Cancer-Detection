import variables
import operators
import glob
import pandas
import ntpath
import numpy
import cv2
import os
import SimpleITK 
import shutil
import random
import math
import multiprocing
from bs4 import BeautifulSoup 
import os
import scipy.misc
import dicom 
import numpy
import math
from multiprocing import Pool


def mhd_file(patient_id):
    for subject_no in range(variables.LUNA_SUBSET_START_INDEX, 10):
        src_dir = variables.LUNA16_RAW_SRC_DIR + "subset" + str(subject_no) + "/"
        for src_path in glob.glob(src_dir + "*.mhd"):
            if patient_id in src_path:
                return src_path
    return None


def load_xml(xml_path, agreement_threshold=0, only_patient=None, save_nods=False):
    pos_lines = []
    neg_lines = []
    extended_lines = []
    with open(xml_path, 'r') as xml_file:
        markup = xml_file.read()
    xml = BeautifulSoup(markup, features="xml")
    if xml.LidcReadMessage is None:
        return None, None, None
    patient_id = xml.LidcReadMessage.ResponseHeader.SeriesInstanceUid.text

    if only_patient is not None:
        if only_patient != patient_id:
            return None, None, None

    src_path = mhd_file(patient_id)
    if src_path is None:
        return None, None, None

    print(patient_id)
    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    num_z, height, width = img_array.shape        
    origin = numpy.array(itk_img.GetOrigin())      
    spacing = numpy.array(itk_img.GetSpacing())    
    rescale = spacing / variables.TARGET_VOXEL_MM

    reading_sessions = xml.LidcReadMessage.find_all("readingSession")
    for reading_session in reading_sessions:

        nods = reading_session.find_all("unblindedReadnod")
        for nod in nods:
            nod_id = nod.nodID.text

            rois = nod.find_all("roi")
            x_min = y_min = z_min = 999999
            x_max = y_max = z_max = -999999
            if len(rois) < 2:
                continue

            for roi in rois:
                z_pos = float(roi.imageZposition.text)
                z_min = min(z_min, z_pos)
                z_max = max(z_max, z_pos)
                edge_maps = roi.find_all("edgeMap")
                for edge_map in edge_maps:
                    x = int(edge_map.xCoord.text)
                    y = int(edge_map.yCoord.text)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                if x_max == x_min:
                    continue
                if y_max == y_min:
                    continue

            x_diameter = x_max - x_min
            x_center = x_min + x_diameter / 2
            y_diameter = y_max - y_min
            y_center = y_min + y_diameter / 2
            z_diameter = z_max - z_min
            z_center = z_min + z_diameter / 2
            z_center -= origin[2]
            z_center /= spacing[2]

            x_center_perc = round(x_center / img_array.shape[2], 4)
            y_center_perc = round(y_center / img_array.shape[1], 4)
            z_center_perc = round(z_center / img_array.shape[0], 4)
            diameter = max(x_diameter , y_diameter)
            diameter_perc = round(max(x_diameter / img_array.shape[2], y_diameter / img_array.shape[1]), 4)

            if nod.characteristics is None:
                print("!!!!nod:", nod_id, " has no charecteristics")
                continue
            if nod.characteristics.malignancy is None:
                print("!!!!nod:", nod_id, " has no malignacy")
                continue

            malignacy = nod.characteristics.malignancy.text
            sphericiy = nod.characteristics.sphericity.text
            margin = nod.characteristics.margin.text
            spiculation = nod.characteristics.spiculation.text
            texture = nod.characteristics.texture.text
            calcification = nod.characteristics.calcification.text
            internal_structure = nod.characteristics.internalStructure.text
            lobulation = nod.characteristics.lobulation.text
            subtlety = nod.characteristics.subtlety.text

            line = [nod_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, malignacy]
            extended_line = [patient_id, nod_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, malignacy, sphericiy, margin, spiculation, texture, calcification, internal_structure, lobulation, subtlety ]
            pos_lines.append(line)
            extended_lines.append(extended_line)

        nonnods = reading_session.find_all("nonnod")
        for nonnod in nonnods:
            z_center = float(nonnod.imageZposition.text)
            z_center -= origin[2]
            z_center /= spacing[2]
            x_center = int(nonnod.locus.xCoord.text)
            y_center = int(nonnod.locus.yCoord.text)
            nod_id = nonnod.nonnodID.text
            x_center_perc = round(x_center / img_array.shape[2], 4)
            y_center_perc = round(y_center / img_array.shape[1], 4)
            z_center_perc = round(z_center / img_array.shape[0], 4)
            diameter_perc = round(max(6 / img_array.shape[2], 6 / img_array.shape[1]), 4)

            line = [nod_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, 0]
            neg_lines.append(line)

    if agreement_threshold > 1:
        filtered_lines = []
        for pos_line1 in pos_lines:
            id1 = pos_line1[0]
            x1 = pos_line1[1]
            y1 = pos_line1[2]
            z1 = pos_line1[3]
            d1 = pos_line1[4]
            overlaps = 0
            for pos_line2 in pos_lines:
                id2 = pos_line2[0]
                if id1 == id2:
                    continue
                x2 = pos_line2[1]
                y2 = pos_line2[2]
                z2 = pos_line2[3]
                d2 = pos_line1[4]
                dist = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2) + math.pow(z1 - z2, 2))
                if dist < d1 or dist < d2:
                    overlaps += 1
            if overlaps >= agreement_threshold:
                filtered_lines.append(pos_line1)


        pos_lines = filtered_lines

    df_annos = pandas.DataFrame(pos_lines, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    df_annos.to_csv(variables.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/" + patient_id + "_annos_pos_lidc.csv", index=False)
    df_neg_annos = pandas.DataFrame(neg_lines, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    df_neg_annos.to_csv(variables.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/" + patient_id + "_annos_neg_lidc.csv", index=False)

    return pos_lines, neg_lines, extended_lines


def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def proc_image(src_path):
    patient_id = ntpath.basename(src_path).replace(".mhd", "")
    print("Patient: ", patient_id)

    dst_dir = variables.LUNA16_EXTRACTED_IMAGE_DIR + patient_id + "/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)

    origin = numpy.array(itk_img.GetOrigin())     
    print("Origin (x,y,z): ", origin)

    direction = numpy.array(itk_img.GetDirection())     
    print("Direction: ", direction)


    spacing = numpy.array(itk_img.GetSpacing())    
    print("Spacing (x,y,z): ", spacing)
    rescale = spacing / variables.TARGET_VOXEL_MM
    print("Rescale: ", rescale)

    img_array = operators.rescale_patient_imgs(img_array, spacing, variables.TARGET_VOXEL_MM)

    img_list = []
    for i in range(img_array.shape[0]):
        img = img_array[i]
        seg_img, mask = operators.seg_lungs(img.copy())
        img_list.append(seg_img)
        img = normalize(img)
        cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_i.png", img * 255)
        cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_m.png", mask * 255)


def proc_pos_annotations_patient(src_path, patient_id):
    df_node = pandas.read_csv("/media/pikachu/Seagate Backup Plus Drive/LC nod Detection/resources/luna16_annotations/annotations.csv")
    dst_dir = variables.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    dst_dir = dst_dir + patient_id + "/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)
    df_patient = df_node[df_node["seriesuid"] == patient_id]
    print("Annos: ", len(df_patient))

    num_z, height, width = img_array.shape        
    origin = numpy.array(itk_img.GetOrigin())     
    print("Origin (x,y,z): ", origin)
    spacing = numpy.array(itk_img.GetSpacing())    
    print("Spacing (x,y,z): ", spacing)
    rescale = spacing /variables.TARGET_VOXEL_MM
    print("Rescale: ", rescale)

    direction = numpy.array(itk_img.GetDirection())      
    print("Direction: ", direction)
    flip_direction_x = False
    flip_direction_y = False
    if round(direction[0]) == -1:
        origin[0] *= -1
        direction[0] = 1
        flip_direction_x = True
        print("Swappint x origin")
    if round(direction[4]) == -1:
        origin[1] *= -1
        direction[4] = 1
        flip_direction_y = True


    assert abs(sum(direction) - 3) < 0.01

    patient_imgs = operators.load_patient_imgs(patient_id, variables.LUNA16_EXTRACTED_IMAGE_DIR, "*_i.png")

    pos_annos = []
    df_patient = df_node[df_node["seriesuid"] == patient_id]
    anno_index = 0
    for index, annotation in df_patient.iterrows():
        node_x = annotation["coordX"]
        if flip_direction_x:
            node_x *= -1
        node_y = annotation["coordY"]
        if flip_direction_y:
            node_y *= -1
        node_z = annotation["coordZ"]
        diam_mm = annotation["diameter_mm"]

        center_float = numpy.array([node_x, node_y, node_z])
        center_int = numpy.rint((center_float-origin) / spacing)



        center_float_rescaled = (center_float - origin) / variables.TARGET_VOXEL_MM
        center_float_percent = center_float_rescaled / patient_imgs.swapaxes(0, 2).shape


        diameter_pixels = diam_mm / variables.TARGET_VOXEL_MM
        diameter_percent = diameter_pixels / float(patient_imgs.shape[1])

        pos_annos.append([anno_index, round(center_float_percent[0], 4), round(center_float_percent[1], 4), round(center_float_percent[2], 4), round(diameter_percent, 4), 1])
        anno_index += 1

    df_annos = pandas.DataFrame(pos_annos, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    df_annos.to_csv(variables.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/" + patient_id + "_annos_pos.csv", index=False)
    return [patient_id, spacing[0], spacing[1], spacing[2]]


def proc_excluded_annotations_patient(src_path, patient_id):
    df_node = pandas.read_csv("/media/pikachu/Seagate Backup Plus Drive/LC nod Detection/resources/luna16_annotations/annotations_excluded.csv")
    dst_dir = variables.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    dst_dir = dst_dir + patient_id + "/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)


    pos_annos_df = pandas.read_csv(variables.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/" + patient_id + "_annos_pos.csv")
    pos_annos_manual = None
    manual_path = variables.EXTRA_DATA_DIR + "luna16_manual_labels/" + patient_id + ".csv"
    if os.path.exists(manual_path):
        pos_annos_manual = pandas.read_csv(manual_path)
        dmm = pos_annos_manual["dmm"]  

    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)
    df_patient = df_node[df_node["seriesuid"] == patient_id]
    print("Annos: ", len(df_patient))

    num_z, height, width = img_array.shape        
    origin = numpy.array(itk_img.GetOrigin())     
    print("Origin (x,y,z): ", origin)
    spacing = numpy.array(itk_img.GetSpacing())   
    print("Spacing (x,y,z): ", spacing)
    rescale = spacing / variables.TARGET_VOXEL_MM
    print("Rescale: ", rescale)

    direction = numpy.array(itk_img.GetDirection())     
    print("Direction: ", direction)
    flip_direction_x = False
    flip_direction_y = False
    if round(direction[0]) == -1:
        origin[0] *= -1
        direction[0] = 1
        flip_direction_x = True
        print("Swappint x origin")
    if round(direction[4]) == -1:
        origin[1] *= -1
        direction[4] = 1
        flip_direction_y = True
        print("Swappint y origin")
    print("Direction: ", direction)
    assert abs(sum(direction) - 3) < 0.01

    patient_imgs = operators.load_patient_imgs(patient_id, variables.LUNA16_EXTRACTED_IMAGE_DIR, "*_i.png")

    neg_annos = []
    df_patient = df_node[df_node["seriesuid"] == patient_id]
    anno_index = 0
    for index, annotation in df_patient.iterrows():
        node_x = annotation["coordX"]
        if flip_direction_x:
            node_x *= -1
        node_y = annotation["coordY"]
        if flip_direction_y:
            node_y *= -1
        node_z = annotation["coordZ"]
        center_float = numpy.array([node_x, node_y, node_z])
        center_int = numpy.rint((center_float-origin) / spacing)
        center_float_rescaled = (center_float - origin) / variables.TARGET_VOXEL_MM
        center_float_percent = center_float_rescaled / patient_imgs.swapaxes(0, 2).shape

        diameter_pixels = 6 / variables.TARGET_VOXEL_MM
        diameter_percent = diameter_pixels / float(patient_imgs.shape[1])

        ok = True

        for index, row in pos_annos_df.iterrows():
            pos_coord_x = row["coord_x"] * patient_imgs.shape[2]
            pos_coord_y = row["coord_y"] * patient_imgs.shape[1]
            pos_coord_z = row["coord_z"] * patient_imgs.shape[0]
            diameter = row["diameter"] * patient_imgs.shape[2]
            print((pos_coord_x, pos_coord_y, pos_coord_z))
            print(center_float_rescaled)
            dist = math.sqrt(math.pow(pos_coord_x - center_float_rescaled[0], 2) + math.pow(pos_coord_y - center_float_rescaled[1], 2) + math.pow(pos_coord_z - center_float_rescaled[2], 2))
            if dist < (diameter + 64): 
                ok = False
                print("CANNOT", center_float_rescaled)
                break

        if pos_annos_manual is not None and ok:
            for index, row in pos_annos_manual.iterrows():
                pos_coord_x = row["x"] * patient_imgs.shape[2]
                pos_coord_y = row["y"] * patient_imgs.shape[1]
                pos_coord_z = row["z"] * patient_imgs.shape[0]
                diameter = row["d"] * patient_imgs.shape[2]
                print((pos_coord_x, pos_coord_y, pos_coord_z))
                print(center_float_rescaled)
                dist = math.sqrt(math.pow(pos_coord_x - center_float_rescaled[0], 2) + math.pow(pos_coord_y - center_float_rescaled[1], 2) + math.pow(pos_coord_z - center_float_rescaled[2], 2))
                if dist < (diameter + 72):  
                    ok = False
                    print("CANNOT", center_float_rescaled)
                    break

        if not ok:
            continue

        neg_annos.append([anno_index, round(center_float_percent[0], 4), round(center_float_percent[1], 4), round(center_float_percent[2], 4), round(diameter_percent, 4), 1])
        anno_index += 1

    df_annos = pandas.DataFrame(neg_annos, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    df_annos.to_csv(variables.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/" + patient_id + "_annos_excluded.csv", index=False)
    return [patient_id, spacing[0], spacing[1], spacing[2]]


def proc_luna_candidates_patient(src_path, patient_id):
    dst_dir = variables.LUNA16_EXTRACTED_IMAGE_DIR + "/_labels/"
    img_dir = dst_dir + patient_id + "/"
    df_pos_annos = pandas.read_csv(dst_dir + patient_id + "_annos_pos_lidc.csv")
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    pos_annos_manual = None
    manual_path = variables.EXTRA_DATA_DIR + "luna16_manual_labels/" + patient_id + ".csv"
    if os.path.exists(manual_path):
        pos_annos_manual = pandas.read_csv(manual_path)

    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)
    print("Pos annos: ", len(df_pos_annos))

    num_z, height, width = img_array.shape      
    origin = numpy.array(itk_img.GetOrigin())     
    print("Origin (x,y,z): ", origin)
    spacing = numpy.array(itk_img.GetSpacing())   
    print("Spacing (x,y,z): ", spacing)
    rescale = spacing / variables.TARGET_VOXEL_MM
    print("Rescale: ", rescale)

    direction = numpy.array(itk_img.GetDirection())     
    print("Direction: ", direction)
    flip_direction_x = False
    flip_direction_y = False
    if round(direction[0]) == -1:
        origin[0] *= -1
        direction[0] = 1
        flip_direction_x = True
        print("Swappint x origin")
    if round(direction[4]) == -1:
        origin[1] *= -1
        direction[4] = 1
        flip_direction_y = True
        print("Swappint y origin")
    print("Direction: ", direction)
    assert abs(sum(direction) - 3) < 0.01

    src_df = pandas.read_csv("/media/pikachu/Seagate Backup Plus Drive/LC nod Detection/resources/luna16_annotations/" + "candidates_V2.csv")
    src_df = src_df[src_df["seriesuid"] == patient_id]
    src_df = src_df[src_df["class"] == 0]
    patient_imgs = operators.load_patient_imgs(patient_id, variables.LUNA16_EXTRACTED_IMAGE_DIR, "*_i.png")
    candidate_list = []

    for df_index, candiate_row in src_df.iterrows():
        node_x = candiate_row["coordX"]
        if flip_direction_x:
            node_x *= -1
        node_y = candiate_row["coordY"]
        if flip_direction_y:
            node_y *= -1
        node_z = candiate_row["coordZ"]
        candidate_diameter = 6
        center_float = numpy.array([node_x, node_y, node_z])
        center_int = numpy.rint((center_float-origin) / spacing)
        center_float_rescaled = (center_float - origin) / variables.TARGET_VOXEL_MM
        center_float_percent = center_float_rescaled / patient_imgs.swapaxes(0, 2).shape
        coord_x = center_float_rescaled[0]
        coord_y = center_float_rescaled[1]
        coord_z = center_float_rescaled[2]

        ok = True

        for index, row in df_pos_annos.iterrows():
            pos_coord_x = row["coord_x"] * patient_imgs.shape[2]
            pos_coord_y = row["coord_y"] * patient_imgs.shape[1]
            pos_coord_z = row["coord_z"] * patient_imgs.shape[0]
            diameter = row["diameter"] * patient_imgs.shape[2]
            dist = math.sqrt(math.pow(pos_coord_x - coord_x, 2) + math.pow(pos_coord_y - coord_y, 2) + math.pow(pos_coord_z - coord_z, 2))
            if dist < (diameter + 64):  
                ok = False
                print("CANNOT", (coord_x, coord_y, coord_z))
                break

        if pos_annos_manual is not None and ok:
            for index, row in pos_annos_manual.iterrows():
                pos_coord_x = row["x"] * patient_imgs.shape[2]
                pos_coord_y = row["y"] * patient_imgs.shape[1]
                pos_coord_z = row["z"] * patient_imgs.shape[0]
                diameter = row["d"] * patient_imgs.shape[2]
                print((pos_coord_x, pos_coord_y, pos_coord_z))
                print(center_float_rescaled)
                dist = math.sqrt(math.pow(pos_coord_x - center_float_rescaled[0], 2) + math.pow(pos_coord_y - center_float_rescaled[1], 2) + math.pow(pos_coord_z - center_float_rescaled[2], 2))
                if dist < (diameter + 72): 
                    ok = False
                    print("CANNOT", center_float_rescaled)
                    break

        if not ok:
            continue

        candidate_list.append([len(candidate_list), round(center_float_percent[0], 4), round(center_float_percent[1], 4), round(center_float_percent[2], 4), round(candidate_diameter / patient_imgs.shape[0], 4), 0])

    df_candidates = pandas.DataFrame(candidate_list, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    df_candidates.to_csv(dst_dir + patient_id + "_candidates_luna.csv", index=False)


def proc_auto_candidates_patient(src_path, patient_id, sample_count=1000, candidate_type="white"):
    dst_dir = variables.LUNA16_EXTRACTED_IMAGE_DIR + "/_labels/"
    img_dir = variables.LUNA16_EXTRACTED_IMAGE_DIR + patient_id + "/"
    df_pos_annos = pandas.read_csv(dst_dir + patient_id + "_annos_pos_lidc.csv")

    pos_annos_manual = None
    manual_path = variables.EXTRA_DATA_DIR + "luna16_manual_labels/" + patient_id + ".csv"
    if os.path.exists(manual_path):
        pos_annos_manual = pandas.read_csv(manual_path)

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)
    print("Pos annos: ", len(df_pos_annos))

    num_z, height, width = img_array.shape        
    origin = numpy.array(itk_img.GetOrigin())      
    print("Origin (x,y,z): ", origin)
    spacing = numpy.array(itk_img.GetSpacing())    
    print("Spacing (x,y,z): ", spacing)
    rescale = spacing / variables.TARGET_VOXEL_MM
    print("Rescale: ", rescale)

    if candidate_type == "white":
        wildcard = "*_c.png"
    else:
        wildcard = "*_m.png"

    src_files = glob.glob(img_dir + wildcard)
    src_files.sort()
    src_candidate_maps = [cv2.imread(src_file, cv2.IMREAD_GRAYSCALE) for src_file in src_files]

    candidate_list = []
    tries = 0
    while len(candidate_list) < sample_count and tries < 10000:
        tries += 1
        coord_z = int(numpy.random.normal(len(src_files) / 2, len(src_files) / 6))
        coord_z = max(coord_z, 0)
        coord_z = min(coord_z, len(src_files) - 1)
        candidate_map = src_candidate_maps[coord_z]
        if candidate_type == "edge":
            candidate_map = cv2.Canny(candidate_map.copy(), 100, 200)

        non_zero_indices = numpy.nonzero(candidate_map)
        if len(non_zero_indices[0]) == 0:
            continue
        nonzero_index = random.randint(0, len(non_zero_indices[0]) - 1)
        coord_y = non_zero_indices[0][nonzero_index]
        coord_x = non_zero_indices[1][nonzero_index]
        ok = True
        candidate_diameter = 6
        for index, row in df_pos_annos.iterrows():
            pos_coord_x = row["coord_x"] * src_candidate_maps[0].shape[1]
            pos_coord_y = row["coord_y"] * src_candidate_maps[0].shape[0]
            pos_coord_z = row["coord_z"] * len(src_files)
            diameter = row["diameter"] * src_candidate_maps[0].shape[1]
            dist = math.sqrt(math.pow(pos_coord_x - coord_x, 2) + math.pow(pos_coord_y - coord_y, 2) + math.pow(pos_coord_z - coord_z, 2))
            if dist < (diameter + 48): 
                ok = False
                print("# Too close", (coord_x, coord_y, coord_z))
                break

        if pos_annos_manual is not None:
            for index, row in pos_annos_manual.iterrows():
                pos_coord_x = row["x"] * src_candidate_maps[0].shape[1]
                pos_coord_y = row["y"] * src_candidate_maps[0].shape[0]
                pos_coord_z = row["z"] * len(src_files)
                diameter = row["d"] * src_candidate_maps[0].shape[1]
                dist = math.sqrt(math.pow(pos_coord_x - coord_x, 2) + math.pow(pos_coord_y - coord_y, 2) + math.pow(pos_coord_z - coord_z, 2))
                if dist < (diameter + 72):  
                    ok = False
                    print("#Too close",  (coord_x, coord_y, coord_z))
                    break

        if not ok:
            continue


        perc_x = round(coord_x / src_candidate_maps[coord_z].shape[1], 4)
        perc_y = round(coord_y / src_candidate_maps[coord_z].shape[0], 4)
        perc_z = round(coord_z / len(src_files), 4)
        candidate_list.append([len(candidate_list), perc_x, perc_y, perc_z, round(candidate_diameter / src_candidate_maps[coord_z].shape[1], 4), 0])

    if tries > 1000:
        print("NOPE")
    df_candidates = pandas.DataFrame(candidate_list, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    df_candidates.to_csv(dst_dir + patient_id + "_candidates_" + candidate_type + ".csv", index=False)


def proc_imgs(delete_existing=False, only_proc_patient=None):
    if delete_existing and os.path.exists(variables.LUNA16_EXTRACTED_IMAGE_DIR):
        print("Removing old stuff..")
        if os.path.exists(variables.LUNA16_EXTRACTED_IMAGE_DIR):
            shutil.rmtree(variables.LUNA16_EXTRACTED_IMAGE_DIR)

    if not os.path.exists(variables.LUNA16_EXTRACTED_IMAGE_DIR):
        os.mkdir(variables.LUNA16_EXTRACTED_IMAGE_DIR)
        os.mkdir(variables.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/")

    for subject_no in range(variables.LUNA_SUBSET_START_INDEX, 10):
        src_dir = variables.LUNA16_RAW_SRC_DIR + "subset" + str(subject_no) + "/"
        src_paths = glob.glob(src_dir + "*.mhd")

        if only_proc_patient is None and True:
            pool = multiprocessing.Pool(variables.WORKER_POOL_SIZE)
            pool.map(proc_image, src_paths)
        else:
            for src_path in src_paths:
                print(src_path)
                if only_proc_patient is not None:
                    if only_proc_patient not in src_path:
                        continue
                proc_image(src_path)


def proc_pos_annotations_patient2():
    candidate_index = 0
    only_patient = None
    for subject_no in range(variables.LUNA_SUBSET_START_INDEX, 10):
        src_dir = variables.LUNA16_RAW_SRC_DIR + "subset" + str(subject_no) + "/"
        for src_path in glob.glob(src_dir + "*.mhd"):
            if only_patient is not None and only_patient not in src_path:
                continue
            patient_id = ntpath.basename(src_path).replace(".mhd", "")
            print(candidate_index, " patient: ", patient_id)
            proc_pos_annotations_patient(src_path, patient_id)
            candidate_index += 1


def proc_excluded_annotations_patients(only_patient=None):
    candidate_index = 0
    for subject_no in range(variables.LUNA_SUBSET_START_INDEX, 10):
        src_dir = variables.LUNA16_RAW_SRC_DIR + "subset" + str(subject_no) + "/"
        for src_path in glob.glob(src_dir + "*.mhd"):
            if only_patient is not None and only_patient not in src_path:
                continue
            patient_id = ntpath.basename(src_path).replace(".mhd", "")
            print(candidate_index, " patient: ", patient_id)
            proc_excluded_annotations_patient(src_path, patient_id)
            candidate_index += 1


def proc_auto_candidates_patients():
    for subject_no in range(variables.LUNA_SUBSET_START_INDEX, 10):
        src_dir = variables.LUNA16_RAW_SRC_DIR + "subset" + str(subject_no) + "/"
        for patient_index, src_path in enumerate(glob.glob(src_dir + "*.mhd")):
            patient_id = ntpath.basename(src_path).replace(".mhd", "")
            print("Patient: ", patient_index, " ", patient_id)
            proc_auto_candidates_patient(src_path, patient_id, sample_count=200, candidate_type="edge")


def proc_luna_candidates_patients(only_patient_id=None):
    for subject_no in range(variables.LUNA_SUBSET_START_INDEX, 10):
        src_dir = variables.LUNA16_RAW_SRC_DIR + "subset" + str(subject_no) + "/"
        for patient_index, src_path in enumerate(glob.glob(src_dir + "*.mhd")):
            patient_id = ntpath.basename(src_path).replace(".mhd", "")
            if only_patient_id is not None and patient_id != only_patient_id:
                continue
            print("Patient: ", patient_index, " ", patient_id)
            proc_luna_candidates_patient(src_path, patient_id)


def proc_lidc_annotations(only_patient=None, agreement_threshold=0):
    file_no = 0
    pos_count = 0
    neg_count = 0
    all_lines = []
    for anno_dir in [d for d in glob.glob("/media/pikachu/Seagate Backup Plus Drive/LC nod Detection/resources/luna16_annotations/*") if os.path.isdir(d)]:
        xml_paths = glob.glob(anno_dir + "/*.xml")
        for xml_path in xml_paths:
            print(file_no, ": ",  xml_path)
            pos, neg, extended = load_xml(xml_path=xml_path, only_patient=only_patient, agreement_threshold=agreement_threshold)
            if pos is not None:
                pos_count += len(pos)
                neg_count += len(neg)
                print("Pos: ", pos_count, " Neg: ", neg_count)
                file_no += 1
                all_lines += extended
    
    df_annos = pandas.DataFrame(all_lines, columns=["patient_id", "anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore", "sphericiy", "margin", "spiculation", "texture", "calcification", "internal_structure", "lobulation", "subtlety"])
    df_annos.to_csv(variables.BASE_DIR_SSD + "lidc_annotations.csv", index=False)

def pipeline_main_luna():
    only_proc_patient = None
    proc_imgs(delete_existing=False, only_proc_patient=only_proc_patient)
    proc_lidc_annotations(only_patient=None, agreement_threshold=0)
    proc_pos_annotations_patient2()
    proc_excluded_annotations_patients(only_patient=None)
    proc_luna_candidates_patients(only_patient_id=None)
    proc_auto_candidates_patients()



def load_patient(src_dir):
    slices = [dicom.read_file(s) for s in glob.glob(src_dir + "*.dcm")]
    print(slices)
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = numpy.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = numpy.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu(slices):
    image = numpy.stack([s.pixel_array for s in slices])
    image = image.astype(numpy.int16)
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(numpy.float64)
            image[slice_number] = image[slice_number].astype(numpy.int16)
        image[slice_number] += numpy.int16(intercept)

    return numpy.array(image, dtype=numpy.int16)


def resample(image, scan, new_spacing=[1, 1, 1]):
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = numpy.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = numpy.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    return image, new_spacing

def cv_flip(img,cols,rows,degree):
    M = cv2.getRotationMatrix2D((cols / 2, rows /2), degree, 1.0)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def extract_dicom_imgs_patient(src_dir):
    target_dir = variables.NDSB3_EXTRACTED_IMAGE_DIR
    print("Dir: ", src_dir)
    dir_path = variables.NDSB3_RAW_SRC_DIR + src_dir + "/"
    patient_id = src_dir
    slices = load_patient(dir_path)
    print(len(slices), "\t", slices[0].SliceThickness, "\t", slices[0].PixelSpacing)
    print("Orientation: ", slices[0].ImageOrientationPatient)

    cos_value = (slices[0].ImageOrientationPatient[0])
    cos_degree = round(math.degrees(math.acos(cos_value)),2)
    
    pixels = get_pixels_hu(slices)
    image = pixels
    print(image.shape)

    invert_order = slices[1].ImagePositionPatient[2] > slices[0].ImagePositionPatient[2]

    pixel_spacing = slices[0].PixelSpacing
    pixel_spacing.append(slices[0].SliceThickness)
    image = operators.rescale_patient_imgs(image, pixel_spacing, variables.TARGET_VOXEL_MM)
    if not invert_order:
        image = numpy.flipud(image)

    for i in range(image.shape[0]):
        patient_dir = target_dir + patient_id + "/"
        if not os.path.exists(patient_dir):
            os.mkdir(patient_dir)
        img_path = patient_dir + "img_" + str(i).rjust(4, '0') + "_i.png"
        org_img = image[i]

        if cos_degree>0.0:
            org_img = cv_flip(org_img,org_img.shape[1],org_img.shape[0],cos_degree)
        img, mask = operators.seg_lungs(org_img.copy())
        org_img = operators.normalize_hu(org_img)
        cv2.imwrite(img_path, org_img * 255)
        cv2.imwrite(img_path.replace("_i.png", "_m.png"), mask * 255)


def pipeline_main_nsdb(clean_targetdir_first=False, only_patient_id=None):

    target_dir = variables.NDSB3_EXTRACTED_IMAGE_DIR
    if clean_targetdir_first and only_patient_id is not None:
        for file_path in glob.glob(target_dir + "*.*"):
            os.remove(file_path)

    if only_patient_id is None:
        dirs = os.listdir(variables.NDSB3_RAW_SRC_DIR)
        for dir_entry in dirs:
            extract_dicom_imgs_patient(dir_entry)
    else:
        extract_dicom_imgs_patient(only_patient_id)


def save_cube_img(target_path, cube_img, rows, cols):
    assert rows * cols == cube_img.shape[0]
    img_height = cube_img.shape[1]
    img_width = cube_img.shape[1]
    res_img = numpy.zeros((rows * img_height, cols * img_width), dtype=numpy.uint8)

    for row in range(rows):
        for col in range(cols):
            target_y = row * img_height
            target_x = col * img_width
            res_img[target_y:target_y + img_height, target_x:target_x + img_width] = cube_img[row * cols + col]

    cv2.imwrite(target_path, res_img)


def get_cube_from_img(img3d, center_x, center_y, center_z, block_size):
    start_x = max(center_x - block_size / 2, 0)
    if start_x + block_size > img3d.shape[2]:
        start_x = img3d.shape[2] - block_size

    start_y = max(center_y - block_size / 2, 0)
    start_z = max(center_z - block_size / 2, 0)
    if start_z + block_size > img3d.shape[0]:
        start_z = img3d.shape[0] - block_size
    start_z = int(start_z)
    start_y = int(start_y)
    start_x = int(start_x)
    res = img3d[start_z:start_z + block_size, start_y:start_y + block_size, start_x:start_x + block_size]
    return res


def make_pos_annotation_imgs():
    src_dir = variables.LUNA_16_trn_DIR2D2 + "metadata/"
    dst_dir = variables.BASE_DIR_SSD + "luna16_trn_cubes_pos/"
    for file_path in glob.glob(dst_dir + "*.*"):
        os.remove(file_path)

    for patient_index, csv_file in enumerate(glob.glob(src_dir + "*_annos_pos.csv")):
        patient_id = ntpath.basename(csv_file).replace("_annos_pos.csv", "")
        df_annos = pandas.read_csv(csv_file)
        if len(df_annos) == 0:
            continue
        imgs = operators.load_patient_imgs(patient_id, variables.LUNA_16_trn_DIR2D2, "*" + "_i" + ".png")

        for index, row in df_annos.iterrows():
            coord_x = int(row["coord_x"] * imgs.shape[2])
            coord_y = int(row["coord_y"] * imgs.shape[1])
            coord_z = int(row["coord_z"] * imgs.shape[0])
            diam_mm = int(row["diameter"] * imgs.shape[2])
            anno_index = int(row["anno_index"])
            cube_img = get_cube_from_img(imgs, coord_x, coord_y, coord_z, 64)
            if cube_img.sum() < 5:
                print(" ***** Skipping ", coord_x, coord_y, coord_z)
                continue

            if cube_img.mean() < 10:
                print(" ***** Suspicious ", coord_x, coord_y, coord_z)

            save_cube_img(dst_dir + patient_id + "_" + str(anno_index) + "_" + str(diam_mm) + "_1_" + "pos.png", cube_img, 8, 8)
        operators.print_tabbed([patient_index, patient_id, len(df_annos)], [5, 64, 8])


def make_annotation_imgs_lidc():
    src_dir = variables.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/"

    dst_dir = variables.BASE_DIR_SSD + "generated_trndata/luna16_trn_cubes_lidc/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for file_path in glob.glob(dst_dir + "*.*"):
        os.remove(file_path)

    for patient_index, csv_file in enumerate(glob.glob(src_dir + "*_annos_pos_lidc.csv")):
        patient_id = ntpath.basename(csv_file).replace("_annos_pos_lidc.csv", "")
        df_annos = pandas.read_csv(csv_file)
        if len(df_annos) == 0:
            continue
        imgs = operators.load_patient_imgs(patient_id, variables.LUNA16_EXTRACTED_IMAGE_DIR, "*" + "_i" + ".png")

        for index, row in df_annos.iterrows():
            coord_x = int(row["coord_x"] * imgs.shape[2])
            coord_y = int(row["coord_y"] * imgs.shape[1])
            coord_z = int(row["coord_z"] * imgs.shape[0])
            malscore = int(row["malscore"])
            anno_index = row["anno_index"]
            anno_index = str(anno_index).replace(" ", "xspacex").replace(".", "xpointx").replace("_", "xunderscorex")
            cube_img = get_cube_from_img(imgs, coord_x, coord_y, coord_z, 64)
            if cube_img.sum() < 5:
                print(" ***** Skipping ", coord_x, coord_y, coord_z)
                continue

            if cube_img.mean() < 10:
                print(" ***** Suspicious ", coord_x, coord_y, coord_z)

            if cube_img.shape != (64, 64, 64):
                print(" ***** incorrect shape !!! ", str(anno_index), " - ",(coord_x, coord_y, coord_z))
                continue

            save_cube_img(dst_dir + patient_id + "_" + str(anno_index) + "_" + str(malscore * malscore) + "_1_pos.png", cube_img, 8, 8)
        operators.print_tabbed([patient_index, patient_id, len(df_annos)], [5, 64, 8])


def make_pos_annotation_imgs_manual():
    src_dir = "resources/luna16_manual_labels/"

    dst_dir = variables.BASE_DIR_SSD + "generated_trndata/luna16_trn_cubes_manual/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for file_path in glob.glob(dst_dir + "*_manual.*"):
        os.remove(file_path)

    for patient_index, csv_file in enumerate(glob.glob(src_dir + "*.csv")):
        patient_id = ntpath.basename(csv_file).replace(".csv", "")
        if "1.3.6.1.4" not in patient_id:
            continue

        print(patient_id)
        df_annos = pandas.read_csv(csv_file)
        if len(df_annos) == 0:
            continue
        imgs = operators.load_patient_imgs(patient_id, variables.LUNA16_EXTRACTED_IMAGE_DIR, "*" + "_i" + ".png")

        for index, row in df_annos.iterrows():
            coord_x = int(row["x"] * imgs.shape[2])
            coord_y = int(row["y"] * imgs.shape[1])
            coord_z = int(row["z"] * imgs.shape[0])
            diameter = int(row["d"] * imgs.shape[2])
            node_type = int(row["id"])
            malscore = int(diameter)
            malscore = min(25, malscore)
            malscore = max(16, malscore)
            anno_index = index
            cube_img = get_cube_from_img(imgs, coord_x, coord_y, coord_z, 64)
            if cube_img.sum() < 5:
                print(" ***** Skipping ", coord_x, coord_y, coord_z)
                continue

            if cube_img.mean() < 10:
                print(" ***** Suspicious ", coord_x, coord_y, coord_z)

            if cube_img.shape != (64, 64, 64):
                print(" ***** incorrect shape !!! ", str(anno_index), " - ",(coord_x, coord_y, coord_z))
                continue

            save_cube_img(dst_dir + patient_id + "_" + str(anno_index) + "_" + str(malscore) + "_1_" + ("pos" if node_type == 0 else "neg") + ".png", cube_img, 8, 8)
        operators.print_tabbed([patient_index, patient_id, len(df_annos)], [5, 64, 8])


def make_candidate_auto_imgs(candidate_types=[]):
    dst_dir = variables.BASE_DIR_SSD + "generated_trndata/luna16_trn_cubes_auto/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for candidate_type in candidate_types:
        for file_path in glob.glob(dst_dir + "*_" + candidate_type + ".png"):
            os.remove(file_path)

    for candidate_type in candidate_types:
        if candidate_type == "falsepos":
            src_dir = "resources/luna16_falsepos_labels/"
        else:
            src_dir = variables.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/"

        for index, csv_file in enumerate(glob.glob(src_dir + "*_candidates_" + candidate_type + ".csv")):
            patient_id = ntpath.basename(csv_file).replace("_candidates_" + candidate_type + ".csv", "")
            print(index, ",patient: ", patient_id, " type:", candidate_type)
            df_annos = pandas.read_csv(csv_file)
            if len(df_annos) == 0:
                continue
            imgs = operators.load_patient_imgs(patient_id, variables.LUNA16_EXTRACTED_IMAGE_DIR, "*" + "_i" + ".png", exclude_wildcards=[])

            row_no = 0
            for index, row in df_annos.iterrows():
                coord_x = int(row["coord_x"] * imgs.shape[2])
                coord_y = int(row["coord_y"] * imgs.shape[1])
                coord_z = int(row["coord_z"] * imgs.shape[0])
                anno_index = int(row["anno_index"])
                cube_img = get_cube_from_img(imgs, coord_x, coord_y, coord_z, 48)
                if cube_img.sum() < 10:
                    print("Skipping ", coord_x, coord_y, coord_z)
                    continue
                try:
                    save_cube_img(dst_dir + patient_id + "_" + str(anno_index) + "_0_" + candidate_type + ".png", cube_img, 6, 8)
                except Exception as ex:
                    print(ex)

                row_no += 1
                max_item = 240 if candidate_type == "white" else 200
                if candidate_type == "luna":
                    max_item = 500
                if row_no > max_item:
                    break


def make_pos_annotation_imgs_manual_ndsb3():
    src_dir = "resources/ndsb3_manual_labels/"
    dst_dir = variables.BASE_DIR_SSD + "generated_trndata/ndsb3_trn_cubes_manual/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)


    trn_label_df = pandas.read_csv("resources/stage1_labels.csv")
    trn_label_df.set_index(["id"], inplace=True)
    for file_path in glob.glob(dst_dir + "*.*"):
        os.remove(file_path)

    for patient_index, csv_file in enumerate(glob.glob(src_dir + "*.csv")):
        patient_id = ntpath.basename(csv_file).replace(".csv", "")
        if "1.3.6.1.4.1" in patient_id:
            continue

        cancer_label = trn_label_df.loc[patient_id]["cancer"]
        df_annos = pandas.read_csv(csv_file)
        if len(df_annos) == 0:
            continue
        imgs = operators.load_patient_imgs(patient_id, variables.NDSB3_EXTRACTED_IMAGE_DIR, "*" + "_i" + ".png")

        anno_index = 0
        for index, row in df_annos.iterrows():
            pos_neg = "pos" if row["id"] == 0 else "neg"
            coord_x = int(row["x"] * imgs.shape[2])
            coord_y = int(row["y"] * imgs.shape[1])
            coord_z = int(row["z"] * imgs.shape[0])
            malscore = int(round(row["dmm"]))
            anno_index += 1
            cube_img = get_cube_from_img(imgs, coord_x, coord_y, coord_z, 64)
            if cube_img.sum() < 5:
                print(" ***** Skipping ", coord_x, coord_y, coord_z)
                continue

            if cube_img.mean() < 10:
                print(" ***** Suspicious ", coord_x, coord_y, coord_z)

            if cube_img.shape != (64, 64, 64):
                print(" ***** incorrect shape !!! ", str(anno_index), " - ",(coord_x, coord_y, coord_z))
                continue
            print(patient_id)
            assert malscore > 0 or pos_neg == "neg"
            save_cube_img(dst_dir + "ndsb3manual_" + patient_id + "_" + str(anno_index) + "_" + pos_neg + "_" + str(cancer_label) + "_" + str(malscore) + "_1_pn.png", cube_img, 8, 8)
        operators.print_tabbed([patient_index, patient_id, len(df_annos)], [5, 64, 8])

def pipeline_main_cubes():
    if not os.path.exists(variables.BASE_DIR_SSD + "generated_trndata/"):
        os.mkdir(variables.BASE_DIR_SSD + "generated_trndata/")

    make_annotation_imgs_lidc()
    make_pos_annotation_imgs_manual()
    make_candidate_auto_imgs(["falsepos", "luna", "edge"])
    make_pos_annotation_imgs_manual_ndsb3() 
##

def pipeline_main():
    pipeline_main_luna()
    pipeline_main_nsdb()
    pipeline_main_cubes()

if __name__ == "__main__":
    pipeline_main()



