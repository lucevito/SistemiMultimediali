import os
import random
import shutil


def copy_file(src_path, des_path, file_names):
    """
    Copies the specified files from the source directory to the destination directory.

    Args:
        src_path (str): Path of the source directory.
        des_path (str): Path of the destination directory.
        file_names (list): List of file names to copy.
    """
    for file in file_names:
        shutil.copyfile(os.path.join(src_path, file), os.path.join(des_path, file))


def split_files(src_dir, train_dir, validation_dir, split_ratio=0.8):
    """
    Splits the files from the source directory between training and validation directories.

    Args:
        src_dir (str): Source directory.
        train_dir (str): Training directory.
        validation_dir (str): Validation directory.
        split_ratio (float): Split ratio between training and validation.
    """
    if len(os.listdir(validation_dir)) == 0:
        split_size = int(len(os.listdir(src_dir)) * split_ratio)
        path_selected = random.sample(os.listdir(src_dir), split_size)
        copy_file(src_dir, train_dir, path_selected)
        rest_path_val = [path for path in os.listdir(src_dir) if path not in path_selected]
        copy_file(src_dir, validation_dir, rest_path_val)


def copyundercondition(directory_main, folder_name, hemoglobin, gender, anemic_eyes_folder, anemic_conjunctive_folder,
                       non_anemic_eyes_folder, non_anemic_conjunctive_folder):
    """
    Copies files based on specific conditions, such as hemoglobin level and gender.

    Args:
        directory_main (str): The main directory containing the source files.
        folder_name (str): The name of the folder from which to copy the files.
        hemoglobin (float): The hemoglobin level used as a condition for copying.
        gender (str): The gender ('M' for male, 'F' for female) used as a condition for copying.
        anemic_eyes_folder (str): The destination folder for anemic eyes files.
        anemic_conjunctive_folder (str): The destination folder for anemic conjunctive files.
        non_anemic_eyes_folder (str): The destination folder for non-anemic eyes files.
        non_anemic_conjunctive_folder (str): The destination folder for non-anemic conjunctive files.
    """
    folder_path = directory_main + "/" + folder_name
    file_list = os.listdir(folder_path)
    file_list.sort()
    if len(file_list) >= 2:
        first_file = file_list[0]
        for file_string in file_list:
            if "_forniceal_palpebral." in file_string:
                forniceal_palpebral = file_string
                if (gender == "M" and hemoglobin < 13) or (gender == "F" and hemoglobin < 12):
                    if (os.path.exists(folder_path + "/" + first_file)
                            and os.path.exists(folder_path + "/" + forniceal_palpebral)):
                        shutil.copy(folder_path + "/" + first_file, anemic_eyes_folder)
                        shutil.copy(folder_path + "/" + forniceal_palpebral, anemic_conjunctive_folder)
                else:
                    if (os.path.exists(folder_path + "/" + first_file)
                            and os.path.exists(folder_path + "/" + forniceal_palpebral)):
                        shutil.copy(folder_path + "/" + first_file, non_anemic_eyes_folder)
                        shutil.copy(folder_path + "/" + forniceal_palpebral, non_anemic_conjunctive_folder)

                break


def sortfile(directory_main, set, df, anemic_eyes_folder, anemic_conjunctive_folder, non_anemic_eyes_folder,
             non_anemic_conjunctive_folder):
    """
    Organizes files into the correct directories based on the DataFrame data.

    Args:
        directory_main (str): Main directory.
        set (str): Name of the dataset.
        df (pd.DataFrame): DataFrame containing data for organizing files.
        anemic_eyes_folder (str): Destination folder for anemic eyes files.
        anemic_conjunctive_folder (str): Destination folder for anemic conjunctive files.
        non_anemic_eyes_folder (str): Destination folder for non-anemic eyes files.
        non_anemic_conjunctive_folder (str): Destination folder for non-anemic conjunctive files.
    """
    for index, row in df.iterrows():
        dataset = row["dataset"]
        gender = row["Sesso"]
        photo1 = row["foto1"]
        photo2 = row["foto2"]
        hemoglobin = float(row["hb"].replace(",", "."))

        subdirectories = [folder_name for folder_name in os.listdir(directory_main) if
                          os.path.isdir(os.path.join(directory_main, folder_name))]
        for folder_name in subdirectories:
            if dataset == set:
                if photo1 in folder_name:
                    copyundercondition(directory_main, folder_name, hemoglobin, gender, anemic_eyes_folder,
                                       anemic_conjunctive_folder, non_anemic_eyes_folder, non_anemic_conjunctive_folder)
                if photo2 in folder_name:
                    copyundercondition(directory_main, folder_name, hemoglobin, gender, anemic_eyes_folder,
                                       anemic_conjunctive_folder, non_anemic_eyes_folder, non_anemic_conjunctive_folder)


def split_files_for_classes(classes_dirs, train_dirs, validation_dirs, split_ratio=0.8):
    """
    Splits files for each class into training and validation sets.

    Args:
        classes_dirs (list): List of directories for each class.
        train_dirs (list): List of directories for training data for each class.
        validation_dirs (list): List of directories for validation data for each class.
        split_ratio (float, optional): Ratio for splitting between training and validation. Default is 0.8.
    """
    for idx, class_dir in enumerate(classes_dirs):
        print('working with:', class_dir)
        split_files(class_dir, train_dirs[idx], validation_dirs[idx], split_ratio)
