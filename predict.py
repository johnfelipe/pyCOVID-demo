import tensorflow as tf
import numpy as np

### error "file does not exist" with TensorFlow 2.1.0 (conda pycovid). Running ok on TF 2.4.1 (conda strmlt)
## doesn't work with full name (best_DenseNet201_Unfreezed--2021-02-23_18-23-58.h5), copied and renamed it to current dir
# load the model
dn201 = tf.keras.models.load_model("D:\\Documents\\GitHub\\DS--pycovid\\streamlit_demo\\model.h5")


###################################### Return model.summary()
def dn201_summary():
    return dn201.summary()


###################################### Pick a random image for each type

def random_pick(diag_type):

    import pandas as pd

    df = pd.read_csv("./src_files/data_test.csv")     # DF of test_set (from train_test_split random_state=123)
    types = ["NORMAL", "COVID", "Viral Pneumonia"]
    datapath = "../data/"     # data root folder


    if diag_type in types:
        # list available numbers for chosen type
        available_numbers = list(np.concatenate(df[df["label"] == diag_type]["filepaths"].str.extract("(\d+)").values.tolist()))

        # choose random file number within available numbers in test set and return corresponding img path
        rand_n = str(np.random.choice(available_numbers, 1)[0])
        img_path = str(datapath + diag_type + "/" + diag_type + " (" + rand_n + ").png")

    return img_path


###################################### Predict !
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def predict_img(file):

    img_width, img_height = 224, 224
    x = load_img(file, target_size=(img_width,img_height))
    x = img_to_array(x)
    x = x / 255
    x = np.expand_dims(x, axis=0)
    array = dn201.predict(x)
    result = array[0]
    #print(result)
    answer = np.argmax(result)
    str_answer = np.where(answer == 0, "Normal", np.where(answer == 1, "COVID-19+", "Viral pneumonia"))
    return str_answer
