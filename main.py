import pandas as pd
import tensorflow as tf
from keras.src.applications import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt
import os
import shutil
import random
import cv2

# Funzione per copiare file da una directory sorgente a una di destinazione
def copy_file(src_path, des_path, file_names):
    """
    Copia i file specificati dalla directory sorgente a quella di destinazione.

    Args:
        src_path (str): Percorso della directory sorgente.
        des_path (str): Percorso della directory di destinazione.
        file_names (list): Elenco dei nomi dei file da copiare.
    """
    for file in file_names:
        shutil.copyfile(os.path.join(src_path, file), os.path.join(des_path, file))

# Funzione per suddividere i file tra directory di addestramento e validazione
def split_files(src_dir, train_dir, validation_dir, split_ration):
    """
    Suddivide i file dalla directory sorgente tra directory di addestramento e validazione.

    Args:
        src_dir (str): Directory sorgente.
        train_dir (str): Directory di addestramento.
        validation_dir (str): Directory di validazione.
        split_ratio (float): Rapporto di divisione tra addestramento e validazione.
    """
    if len(os.listdir(validation_dir)) == 0:
        split_size = int(len(os.listdir(src_dir)) * 0.8)
        path_selected = random.sample(os.listdir(src_dir), split_size)
        copy_file(src_dir, train_dir, path_selected)
        rest_path_val = [path for path in os.listdir(src_dir) if path not in path_selected]
        copy_file(src_dir, validation_dir, rest_path_val)

# Funzione per copiare file in base a condizioni specifiche
def copyfile(directory_principale, nome_cartella, emoglobina, sesso):
    """
    Copia file in base a condizioni specifiche come emoglobina e sesso.

    Args:
        directory_principale (str): Directory principale.
        nome_cartella (str): Nome della cartella da cui copiare i file.
        emoglobina (float): Valore dell'emoglobina.
        sesso (str): Genere (M per maschio, F per femmina).
    """
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

# Funzione per organizzare i file nelle directory corrette
def sortfile(directory_principale, set, df):
    """
    Organizza i file nelle directory corrette in base ai dati del DataFrame.

    Args:
        directory_principale (str): Directory principale.
        set (str): Nome del set di dati.
        df (pd.DataFrame): DataFrame contenente i dati da utilizzare per l'organizzazione dei file.
    """
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

# Funzione per elencare tutte le cartelle foglie in un percorso
def elenca_cartelle_foglie(path):
    """
    Restituisce una lista di tutte le cartelle foglie in un percorso specificato.

    Args:
        path (str): Percorso di ricerca.

    Returns:
        list: Lista delle cartelle foglie nel percorso specificato.
    """
    cartelle_foglie = []

    for root, dirs, files in os.walk(path):
        if not dirs:
            cartelle_foglie.append(root)

    return cartelle_foglie

# Funzione per creare una directory se non esiste già
def create_directory(path):
    """
    Crea una directory se non esiste già.

    Args:
        path (str): Percorso della directory da creare.
    """
    if not os.path.exists(path):
        os.makedirs(path)

# Funzione per convertire immagini e risolvere problemi di formato
def fiximage(data):
    """
    Converte le immagini in un formato specifico e risolve i problemi di formato.

    Args:
        data (str): Percorso della directory contenente le immagini da elaborare.
    """
    for dir in os.listdir(data):
        for dir2 in os.listdir(os.path.join(data, dir)):
            if "congiuntive" in dir2:
                for dir3 in os.listdir(os.path.join(data, dir, dir2)):
                    for file in os.listdir(os.path.join(data, dir, dir2, dir3)):
                        img = cv2.imread(os.path.join(data, dir, dir2, dir3, file), cv2.IMREAD_UNCHANGED)
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                        new_file = file.split(".png")[0] + ".jpg"
                        cv2.imwrite(os.path.join(data, dir, dir2, dir3, file), img)

# Caricamento dei dati da un file CSV
percorso_csv = "database_sclere/hbvalue.csv"

df = pd.read_csv(percorso_csv, sep=';')
df = df.drop(columns=[df.columns[-1]])
print(df)
# Creazione delle directory per l'organizzazione dei dati
datasetcartella = "Dataset"
cartella_occhi = "Dataset/occhi"
cartella_congiuntive = "Dataset/congiuntive "

cartella_anemici_occhi = "Dataset/occhi/anemici"
cartella_non_anemici_occhi = "Dataset/occhi/non_anemici"
artella_anemici_congiuntive = "Dataset/congiuntive/anemici"
cartella_non_anemici_congiuntive = "Dataset/congiuntive/non_anemici"

os.makedirs(datasetcartella, exist_ok=True)
os.makedirs(cartella_occhi, exist_ok=True)
os.makedirs(cartella_congiuntive, exist_ok=True)

os.makedirs(cartella_anemici_occhi, exist_ok=True)
os.makedirs(cartella_non_anemici_occhi, exist_ok=True)
os.makedirs(artella_anemici_congiuntive, exist_ok=True)
os.makedirs(cartella_non_anemici_congiuntive, exist_ok=True)
# Impostazione di una soglia per l'anemia
soglia_anemia = 10
# Organizzazione dei file nelle directory appropriate
directory_principale = "database_sclere/Dataset indiano/"
set = "ind"
sortfile(directory_principale, set, df)
directory_principale = "database_sclere/Italiano congiuntive/Dataset congiuntive italiano segmentato/"
set = "ita"
sortfile(directory_principale, set, df)
# Elenco delle directory foglia contenenti le immagini
directory_path = "dataset"
classes_dirs = elenca_cartelle_foglie(directory_path)
# Creazione della struttura delle directory per l'addestramento e la validazione
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

create_directory(congiuntive_anemici_train_dir)
create_directory(congiuntive_non_anemici_train_dir)
create_directory(occhi_anemici_train_dir)
create_directory(occhi_non_anemici_train_dir)

create_directory(congiuntive_anemici_validation_dir)
create_directory(congiuntive_non_anemici_validation_dir)
create_directory(occhi_anemici_validation_dir)
create_directory(occhi_non_anemici_validation_dir)
# Divisione dei file tra addestramento e validazione
validation_classes = [congiuntive_anemici_validation_dir,
                      congiuntive_non_anemici_validation_dir,
                      occhi_anemici_validation_dir,
                      occhi_non_anemici_validation_dir]

train_classes = [congiuntive_anemici_train_dir,
                 congiuntive_non_anemici_train_dir,
                 occhi_anemici_train_dir,
                 occhi_non_anemici_train_dir]

split_ratio = 0.8
for idx, class_ in enumerate(classes_dirs):
    print('working with:', class_)
    split_files(class_, train_classes[idx], validation_classes[idx], split_ratio)
# Correzione del formato delle immagini
data = "data"
fiximage(data)
# Creazione degli oggetti ImageDataGenerator per addestramento e validazione
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
# Creazione e compilazione del modello di rete neurale
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.summary()
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
# Addestramento del modello
history = model.fit(train_d_gen, validation_data=val_d_gen, epochs=20)
loss = history.history['loss']
val_loss = history.history['val_loss']
# Visualizzazione delle curve di apprendimento
plt.figure(figsize=(12, 6))
plt.plot(loss, 'b', label='training loss')
plt.plot(val_loss, 'orange', label='validation loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(acc, 'b', label='training acc')
plt.plot(val_acc, 'orange', label='val acc')
plt.legend()
plt.show()
# Valutazione del modello sui dati di validazione
print(model.evaluate(val_d_gen))
model.evaluate(val_d_gen)

