import pandas as pd
import os

from copy_operations import sortfile, split_files_for_classes
from directory_operations import create_directory, list_leaf_folders
from image_operations import fiximage, create_train_data_generator, create_validation_data_generator, \
    create_pre_trained_model
from model import create_model, compile_model, train_model, plot_learning_curves, evaluate_model

csv_path = "database_sclere/hbvalue.csv"
indian_dataset_directory = "database_sclere/Dataset indiano/"
italian_dataset_directory = "database_sclere/Italiano congiuntive/"
anemia_threshold = 10

df = pd.read_csv(csv_path, sep=';')
df = df.drop(columns=[df.columns[-1]])
print(df)

dataset_path = os.path.join('Dataset')
eyes_folder = os.path.join('Dataset/eyes')
conjunctive_folder = os.path.join('Dataset/conjunctive')
anemic_eyes_folder = os.path.join('Dataset/eyes/anemic')
non_anemic_eyes_folder = os.path.join('Dataset/eyes/non_anemic')
anemic_conjunctive_folder = os.path.join('Dataset/conjunctive/anemic')
non_anemic_conjunctive_folder = os.path.join('Dataset/conjunctive/non_anemic')

base_dir = os.path.join('data')
# train folder
train_dir = os.path.join(base_dir, 'train')
eyes_train_folder = os.path.join(train_dir, 'eyes')
conjunctive_train_folder = os.path.join(train_dir, 'conjunctive')

anemic_conjunctive_train_folder = os.path.join(conjunctive_train_folder, 'anemic')
non_anemic_conjunctive_train_folder = os.path.join(conjunctive_train_folder, 'non_anemic')
anemic_eyes_train_folder = os.path.join(eyes_train_folder, 'anemic')
non_anemic_eyes_train_folder = os.path.join(eyes_train_folder, 'non_anemic')

# validation folder
validation_dir = os.path.join(base_dir, 'validation')
conjunctive_validation_folder = os.path.join(validation_dir, 'conjunctive')
eyes_validation_folder = os.path.join(validation_dir, 'eyes')

anemic_conjunctive_validation_folder = os.path.join(conjunctive_validation_folder, 'anemic')
non_anemic_conjunctive_validation_folder = os.path.join(conjunctive_validation_folder, 'non_anemic')
anemic_eyes_validation_dir = os.path.join(eyes_validation_folder, 'anemic')
non_anemic_eyes_validation_dir = os.path.join(eyes_validation_folder, 'non_anemic')

create_directory(dataset_path)
create_directory(eyes_folder)
create_directory(conjunctive_folder)
create_directory(anemic_eyes_folder)
create_directory(non_anemic_eyes_folder)
create_directory(anemic_conjunctive_folder)
create_directory(non_anemic_conjunctive_folder)

create_directory(anemic_conjunctive_train_folder)
create_directory(non_anemic_conjunctive_train_folder)
create_directory(anemic_eyes_train_folder)
create_directory(non_anemic_eyes_train_folder)
create_directory(anemic_conjunctive_validation_folder)
create_directory(non_anemic_conjunctive_validation_folder)
create_directory(anemic_eyes_validation_dir)
create_directory(non_anemic_eyes_validation_dir)

sortfile(indian_dataset_directory, "ind", df, anemic_eyes_folder, anemic_conjunctive_folder,
         non_anemic_eyes_folder, non_anemic_conjunctive_folder)
sortfile(italian_dataset_directory, "ita", df, anemic_eyes_folder, anemic_conjunctive_folder,
         non_anemic_eyes_folder, non_anemic_conjunctive_folder)

# File split between training and validation
train_eyes = [anemic_eyes_train_folder, non_anemic_eyes_train_folder]
train_conjunctive = [anemic_conjunctive_train_folder, non_anemic_conjunctive_train_folder]
validation_eyes = [anemic_eyes_validation_dir, non_anemic_eyes_validation_dir]
validation_conjunctive = [anemic_conjunctive_validation_folder, non_anemic_conjunctive_validation_folder]

eyes_classes_dirs = list_leaf_folders(os.path.join(dataset_path, 'eyes'))
conjunctive_classes_dirs = list_leaf_folders(os.path.join(dataset_path, 'conjunctive'))

split_files_for_classes(eyes_classes_dirs, train_eyes, validation_eyes)
split_files_for_classes(conjunctive_classes_dirs, train_conjunctive, validation_conjunctive)

# Image format correction
fiximage(base_dir)

train_e_gen = create_train_data_generator(eyes_train_folder)
val_e_gen = create_validation_data_generator(eyes_validation_folder)
train_c_gen = create_train_data_generator(conjunctive_train_folder)
val_c_gen = create_validation_data_generator(conjunctive_validation_folder)

model_eyes = create_model()
model_eyes = compile_model(model_eyes)
history = train_model(model_eyes, train_e_gen, val_e_gen)
plot_learning_curves(history)
evaluate_model(model_eyes, val_e_gen)

model_conjunctive = create_model()
model_conjunctive = compile_model(model_conjunctive)
history = train_model(model_conjunctive, train_c_gen, val_c_gen)
plot_learning_curves(history)
evaluate_model(model_conjunctive, val_c_gen)

pre_trained_model_eyes = create_pre_trained_model()
pre_trained_model_eyes = compile_model(pre_trained_model_eyes)
history_pre_trained = train_model(pre_trained_model_eyes, train_e_gen, val_e_gen, epochs=50)
plot_learning_curves(history_pre_trained)
evaluate_model(pre_trained_model_eyes, val_e_gen)

pre_trained_model_conjunctive = create_pre_trained_model()
pre_trained_model_conjunctive = compile_model(pre_trained_model_conjunctive)
history_pre_trained = train_model(pre_trained_model_conjunctive, train_e_gen, val_e_gen, epochs=50)
plot_learning_curves(history_pre_trained)
evaluate_model(pre_trained_model_conjunctive, val_e_gen)
