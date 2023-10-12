import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt
import os
import shutil
import random


def copy_file(src_path, des_path, file_names):
    for file in file_names:
        shutil.copyfile(os.path.join(src_path, file), os.path.join(des_path, file))


def split_files(src_dir, train_dir, validation_dir, split_ration):
    if len(os.listdir(validation_dir)) == 0:
        split_size = int(len(os.listdir(src_dir)) * 0.8)
        path_selected = random.sample(os.listdir(src_dir), split_size)
        copy_file(src_dir, train_dir, path_selected)
        rest_path_val = [path for path in os.listdir(src_dir) if path not in path_selected]
        copy_file(src_dir, validation_dir, rest_path_val)


def copyfile(directory_principale, nome_cartella, emoglobina, sesso):
    percorso_cartella = directory_principale + "/" + nome_cartella
    elenco_file = os.listdir(percorso_cartella)
    elenco_file.sort()
    if len(elenco_file) >= 2:
        primo = elenco_file[0]
        for stringa in elenco_file:
            if "_forniceal_palpebral." in stringa:
                forniceal_palpebral = stringa
                if (sesso == "M" and emoglobina < 13) or (sesso == "F" and emoglobina < 12):
                    if (os.path.exists(percorso_cartella + "/" + primo)
                            and os.path.exists(percorso_cartella + "/" + forniceal_palpebral)):
                        shutil.copy(percorso_cartella + "/" + primo, cartella_anemici_occhi)
                        shutil.copy(percorso_cartella + "/" + forniceal_palpebral, artella_anemici_congiuntive)
                else:
                    if (os.path.exists(percorso_cartella + "/" + primo)
                            and os.path.exists(percorso_cartella + "/" + forniceal_palpebral)):
                        shutil.copy(percorso_cartella + "/" + primo, cartella_non_anemici_occhi)
                        shutil.copy(percorso_cartella + "/" + forniceal_palpebral, cartella_non_anemici_congiuntive)

                break


def sortfile(directory_principale, set, df):
    for index, row in df.iterrows():
        dataset = row["dataset"]
        sesso = row["Sesso"]
        foto1 = row["foto1"]
        foto2 = row["foto2"]
        emoglobina = float(row["hb"].replace(",", "."))

        elenco_cartelle = [nome_cartella for nome_cartella in os.listdir(directory_principale) if
                           os.path.isdir(os.path.join(directory_principale, nome_cartella))]
        for nome_cartella in elenco_cartelle:
            if dataset == set:
                if foto1 in nome_cartella:
                    copyfile(directory_principale, nome_cartella, emoglobina, sesso)
                if foto2 in nome_cartella:
                    copyfile(directory_principale, nome_cartella, emoglobina, sesso)


def elenca_cartelle_foglie(path):
    cartelle_foglie = []

    for root, dirs, files in os.walk(path):
        if not dirs:
            cartelle_foglie.append(root)

    return cartelle_foglie


datasetcartella = "Dataset"
cartella_occhi = "Dataset/occhi"
cartella_congiuntive = "Dataset/congiuntive "

cartella_anemici_occhi = "Dataset/occhi/anemici"
cartella_non_anemici_occhi = "Dataset/occhi/non_anemici"
artella_anemici_congiuntive = "Dataset/congiuntive/anemici"
cartella_non_anemici_congiuntive = "Dataset/congiuntive/non_anemici"

percorso_csv = "database_sclere/hbvalue.csv"

df = pd.read_csv(percorso_csv, sep=';')
df = df.drop(columns=[df.columns[-1]])
print(df)

os.makedirs(datasetcartella, exist_ok=True)
os.makedirs(cartella_occhi, exist_ok=True)
os.makedirs(cartella_congiuntive, exist_ok=True)

os.makedirs(cartella_anemici_occhi, exist_ok=True)
os.makedirs(cartella_non_anemici_occhi, exist_ok=True)
os.makedirs(artella_anemici_congiuntive, exist_ok=True)
os.makedirs(cartella_non_anemici_congiuntive, exist_ok=True)

soglia_anemia = 10

directory_principale = "database_sclere/Dataset indiano/"
set = "ind"
sortfile(directory_principale, set, df)
directory_principale = "database_sclere/Italiano congiuntive/Dataset congiuntive italiano segmentato/"
set = "ita"
sortfile(directory_principale, set, df)

directory_path = "dataset"

classes_dirs = elenca_cartelle_foglie(directory_path)

base_dir = os.path.join('data')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

congiuntive_anemici_train_dir = os.path.join(train_dir, 'congiuntive/anemici')
congiuntive_non_anemici_train_dir = os.path.join(train_dir, 'congiuntive/non_nemici')
occhi_anemici_train_dir = os.path.join(train_dir, 'occhi/anemici')
occhi_non_anemici_train_dir = os.path.join(train_dir, 'occhi/non_nemici')

congiuntive_anemici_validation_dir = os.path.join(validation_dir, 'congiuntive/anemici')
congiuntive_non_anemici_validation_dir = os.path.join(validation_dir, 'congiuntive/non_nemici')
occhi_anemici_validation_dir = os.path.join(validation_dir, 'occhi/anemici')
occhi_non_anemici_validation_dir = os.path.join(validation_dir, 'occhi/non_nemici')


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_directory(congiuntive_anemici_train_dir)
create_directory(congiuntive_non_anemici_train_dir)
create_directory(occhi_anemici_train_dir)
create_directory(occhi_non_anemici_train_dir)

create_directory(congiuntive_anemici_validation_dir)
create_directory(congiuntive_non_anemici_validation_dir)
create_directory(occhi_anemici_validation_dir)
create_directory(occhi_non_anemici_validation_dir)

validation_classes = [congiuntive_anemici_validation_dir,
                      congiuntive_non_anemici_validation_dir,
                      occhi_anemici_validation_dir,
                      occhi_non_anemici_validation_dir]

train_classes = [congiuntive_anemici_train_dir,
                 congiuntive_non_anemici_train_dir,
                 occhi_anemici_train_dir,
                 occhi_non_anemici_train_dir]

samples = []
for class_ in classes_dirs:
    paths = os.listdir(class_)
    paths = random.sample(paths, 4)
    samples.append([os.path.join(class_, path) for path in paths])
    # lets flatten samples array
samples = [ele for arr_ in samples for ele in arr_]
print(samples)
fig = plt.gcf()
fig.set_size_inches(20, 20)

from PIL import Image

for idx, ele in enumerate(samples):
    subplot = plt.subplot(5, 4, idx + 1)
    try:
        img = Image.open(ele)
        plt.imshow(img)
    except Exception as e:
        print("Errore nella lettura del file:", e)
plt.show()

split_ratio = 0.8
for idx, class_ in enumerate(classes_dirs):
    print('working with:', class_)
    split_files(class_, train_classes[idx], validation_classes[idx], split_ratio)

train_gen = ImageDataGenerator(rescale=1. / 255.,
                               rotation_range=40,
                               shear_range=.2,
                               width_shift_range=.2,
                               height_shift_range=.2,
                               horizontal_flip=True,
                               vertical_flip=True,
                               zoom_range=0.2,
                               fill_mode='nearest')

train_d_gen = train_gen.flow_from_directory(train_dir, class_mode='categorical', target_size=(150, 150), batch_size=16)

val_gen = ImageDataGenerator(rescale=1. / 255.)
val_d_gen = val_gen.flow_from_directory(validation_dir, class_mode='categorical', target_size=(150, 150), batch_size=2)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.summary()

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

history = model.fit(train_d_gen, validation_data=val_d_gen, epochs=20)
