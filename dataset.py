import pandas as pd
import os.path
import cv2
import matplotlib.pyplot as plt
import numpy as np

# data root folder
datapath = "data/"


#### make dataframe out of the 3 metadata files

def meta_df():

    normal = pd.read_excel(datapath + "NORMAL.metadata.xlsx")
    covid = pd.read_excel(datapath + "COVID.metadata.xlsx")
    viral = pd.read_excel(datapath + "Viral Pneumonia.metadata.xlsx")

    # harmonize covid.FILE NAME nomenclature
    covid["FILE NAME"] = covid["FILE NAME"].str.replace("COVID", "COVID-")

    # merge all
    meta = pd.concat([normal, covid, viral], axis=0)

    ## create "type" and "n_img" cols
    # split string at first "-" into 2 cols
    meta[["type", "n_img"]] = meta["FILE NAME"].str.split("-", n = 1, expand = True)
    meta["n_img"] = meta["n_img"].astype("int")
    meta = meta.reindex(columns=["type", "n_img", "FORMAT", "SIZE", "URL", "FILE NAME"])       # reorder cols
    meta = meta.reset_index(drop=True)                                                        # reset index

    meta.columns = ["type", "n_img", "format", "size", "source_url", "filename"]            # rename cols

    return meta


#### Display a random image for each type

def random_set():
    types = ["NORMAL", "COVID", "Viral Pneumonia"]
    num_files = {}  # number of files in each folder/type
    rand_n = {}  # store random number for each type
    img_paths = []  # filepaths to selected random img
    imgs = []  # corresponding img arrays

    for t in types:
        # number of files in each folder/type
        num_files[t] = len(
            [f for f in os.listdir(datapath + t + "/") if os.path.isfile(os.path.join(datapath + t + "/", f))])

        # choose random file number within range of each t and generate filepath
        rand_n[t] = str(np.random.randint(1, num_files[t], 1)[0])
        img_paths.append(datapath + t + "/" + t + " (" + rand_n[t] + ").png")

    # read imgs in list
    for i in img_paths:
        orig = cv2.imread(i, cv2.IMREAD_GRAYSCALE)  # read img
        resized = cv2.resize(orig, dsize=(256, 256))  # resize it to 256 x 256
        imgs.append(resized)

    # display images
    fig = plt.figure(figsize=(15, 5))

    for i, t in zip(range(3), types):
        ax = fig.add_subplot(131 + i)
        ax.imshow(imgs[i], cmap="gray")
        ax.set_title(t + " (" + rand_n[t] + ")",
                     fontdict={'fontsize': 15,
                               'fontweight': 'bold'})

    return types, num_files, rand_n, img_paths, imgs, fig
