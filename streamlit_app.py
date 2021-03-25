# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 14:59:22 2021

@author: Caro
"""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="pyCOVID", page_icon="./streamlit_imgs/COVID_logo.png", layout="wide")
# st.set_page_config(page_title="pyCOVID", page_icon="./streamlit_imgs/COVID_logo.png", layout="centered")


# st.sidebar.image("./streamlit_imgs/pyCOVID_spacefade.png", width=300)
st.sidebar.image("./streamlit_imgs/pyCOVID_logo_circle.png", width=300)

st.sidebar.title("pyCOVID")
st.sidebar.write("Classification project of COVID+ chest X-rays")


pages = ["Project", "Dataset exploration", "Preprocessing", "Modelization", "Prediction demo", "Improving & understanding the model", "Conclusion & Perspectives"]
section = st.sidebar.radio('', pages)

st.sidebar.write("")

description = """
End of training project &mdash; Promotion Bootcamp Data Scientist Dec. 2020

**Brahim MOUDJED** : [LinkedIn](http://www.linkedin.com/in/brahim-moudjed-lyon-fr)

**Caroline PIROU** : [LinkedIn](http://www.linkedin.com/in/caroline-pirou)

Mentor : Souhail HADGI  
Presented on March 5th, 2021
"""

st.sidebar.info(description)

page_bg_img = '''
    <style>
    body {
    background-image: url("https://i.imgur.com/Fcbc9Mb.png");
    background-size: cover;    
    }
    </style>
    '''

############################################################################################################# Project presentation

if section == "Project":

    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.title("pyCOVID")
    st.header("Classification of COVID-19+, normal and viral pneumonia chest X-rays")


    st.subheader("Context")
    st.write("""Coronavirus disease 2019 (COVID-19) is a contagious disease caused by **severe acute respiratory syndrome coronavirus 2** (SARS-CoV-2).
The first case was identified in Wuhan, China, in December 2019. The disease has since spread worldwide, leading to an **ongoing pandemic** that has caused
almost **115 million reported contaminations** and over **2.5 million deaths worldwide** as of March 1st, 2021.

Amongst the available diagnostic tools is the chest radiography: chest X-rays can detect COVID-19+ patients since its main clinical characteristic is that of bronchopneumonia.""")

# The COVID-19 pandemic has put some **health systems under immense pressure and stretched others beyond their capacity**.
# As such, responding to this public health emergency and successfully minimizing its impact requires every health resource to be leveraged. AI could be one of them.
# Indeed, correctly diagnosing symptomatic patients arriving at the hospital is critical for determining their care protocol, and quarantining if necessary.
# In a time where **health professionals are particularly affected by the disease and under staffing is prevalent, machine-assisted diagnostics could be more efficient**.
# Especially in countries where the reference RT-PCR tests are in significant shortage.

    st.image("./streamlit_imgs/HorryCOVIDXrays.png")
    # st.image("./streamlit_imgs/HorryCOVIDXrays_small.png")
    st.markdown("""<center><small><i>Evolution of COVID-19 patient. Adapted from Horry </i>et al.<i>, 2020, </i>IEEE Access</small></center>""",
                unsafe_allow_html=True)

    st.subheader("AI in diagnostics")
    st.write("""Bai _et al._ studied performances of a Deep Learning model applied to chest CT-scan slices of COVID-19+ and pneumonia patients.
They found the **AI model had higher** test **accuracy**, **sensitivity** and **specificity than radiologists**, and that **radiologists performed better with AI assistance**.
""")

    st.image("./streamlit_imgs/BaiTab5Adapted.png")
    st.markdown(
        """<center><small><i>Adapted from Bai </i>et al.<i>, 2020, </i>Radiology</small></center>""",
        unsafe_allow_html=True)

    st.subheader("Objective")
    st.write("Our goal was to design a model able to **accurately differentiate and classify COVID-19+, healthy or viral pneumonia** patients based on a dataset of **chest X-rays**.")

############################################################################################################# Dataset exploration

from dataset import meta_df, random_set

# meta["type"].unique()    # check content of `type` col
# meta.info()

code = """Google Colab
└───data
    ├───COVID                            # image folder
    ├───NORMAL                           # image folder
    ├───Viral Pneumonia                  # image folder
    │
    ├─ COVID.metadata.xlsx
    ├─ NORMAL.metadata.xlsx
    └─ Viral Pneumonia.metadata.xlsx
"""

data_prez = """The _COVID-19 Radiography Database_ dataset has been made publicly available on [Kaggle](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) by Chowdhury _et al._

It contains **3886 chest X-rays** in `PNG` format, aggregated from various sources, which have been diagnosed as either **COVID-19 positive** (`COVID`), **healthy** (`Normal`), or **viral pneumonia** (`Viral Pneumonia`).
Metadata files have been generated for each class as an Excel table. The dataset exploration revealed that the `COVID-1200.png` file was **missing** from the metadata dataframe while present in the image folder.
   
The `Normal` and `Viral Pneumonia` images are **1024 x 1024px** while `COVID` images are all **256 x 256px**.
"""

if section == "Dataset exploration":
    st.title("Dataset exploration")

    st.write(data_prez)

    st.subheader("Data structure")
    st.code(code)

    meta = meta_df()

    st.write("First and last 3 lines of the **metadata dataframe**:")
    st.dataframe(meta.iloc[pd.np.r_[0:3, -3:0]])

    st.markdown("<br>", unsafe_allow_html=True)

    # with st.beta_expander("Show data description"):
    #     st.dataframe(meta.drop(columns="n_img").groupby("type").describe())
    #
    # st.markdown("<br><br>", unsafe_allow_html=True)

    st.image("./streamlit_imgs/piechart.png")

    st.header("Random X-ray from each class:")

    types, num_files, rand_n, img_paths, imgs, fig = random_set()
    st.pyplot(fig)

    rand_gen = st.button("Generate random set")

    st.header("Differences in contrast")

    contrast_block = """We counted the intensity values taken by all the pixels of all the images for a given type. We can see here that there are **much more 0 values (black pixels)
    in the _Normal_ X-rays** than in the _Viral pneumonia_ X-rays than in the _COVID-19_ X-rays. Distributions seem to be very close. However, it can be noticed that the **distribution of _COVID-19_
    is slightly shifted toward the brighter values** compared to the normal, while the **_Viral pneumonia_ seems to be mostly shifted toward the mid-high values** (150-200).
    
This correlates with the differences in contrast we have observed between the three classes while generating multiple random sets: _Normal_ X-rays have, in general, empty/black lungs,
while _COVID-19_ and _Viral pneumonia_ X-rays have, in general, more opaque/white lungs, indicative of the fluid and debris filling the lungs.
"""

    st.write(contrast_block)

    st.image("./streamlit_imgs/intensity_counts.png")

# Dataset exploration
############################################################################################################# Preprocessing

if section == "Preprocessing":
    st.title("Preprocessing")

    st.header("Resizing")
    st.write("Before being fed to the various models, images were all **resized to 224 x 224 pixels** as it is a commonly used resolution,"
             "empirically determined to be a good compromise between smaller size / computing time, and preservation of important features.")

    st.header("Rescaling")
    st.write(" Our original images consist in RGB coefficients in the 0-255, "
             "but such values would be too high for our models to process (given a typical learning rate), "
             "so we target values between 0 and 1 instead by **scaling with a 1/255. factor**.")

    st.header("Data augmentation")
    augm = """In order to increase the number of images fed to the model, and to avoid overfitting, images were augmented using 3 transformations:
    
* **rotation**: 10°
* **horizontal/vertical shift**: 10% of image size
* **horizontal/vertical flip**
    """

    st.write(augm)

    st.image("./streamlit_imgs/DS_python_keras_picasso_augmentation.png")
    st.markdown(
        """<center><small><i>From DataScientest.</i></small></center>""",
        unsafe_allow_html=True)

    st.header("Train/test split")
    st.write("""The dataset was split in two, in a stratified manner:

* **training set:** 80%
* **testing set:** 20%""")



# Preprocessing
############################################################################################################# Modelization

if section == "Modelization":
    st.title("Modelization")

    model_desc = """Being able to extract various features and patterns from data, some (most ?) not obvious to the human eye, in order to classify them
    is a typical Deep Learning problem that uses large artificial neural networks to solve it.

**C**onvolutional **N**eural **N**etworks are a class of deep neural networks applied to **image** classification. They typically use convolutional layers to extract the features,
 and fully connected layers (dense layers) to classify the images.
    """

    st.write(model_desc)

    st.image("./streamlit_imgs/CNN_archi.png")
    st.markdown("""<center><small><i>From <a href="https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53">
        A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way</a>, by Sumit Saha</i></small></center>""",
                unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    hdrs = ["1. Simple CNN model", "2. Benchmark", "3. Transfer learning: DenseNet201", "4. Recap"]

    header_select = st.selectbox("Navigate:", hdrs)

    st.markdown("<br>", unsafe_allow_html=True)

    if header_select == "1. Simple CNN model":


        st.header("1. Simple CNN model")

        st.write("""We first started by designing a simple CNN model with **2 convolution layers** to extract features and **3 dense layers** to classify images.""")


        st.subheader("Architecture")
        st.image("./streamlit_imgs/customCNN--archi.png")

        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("Training hyperparameters")
        st.write("""We optimized the model using by **minimizing the loss on the validation dataset**. The model was fitted over **20 epochs** and:

1. the **learning rate was reduced** by a factor 10 when the loss of the validation set reached a plateau for 2 epochs,
1. the **fitting stopped** when the validation set loss hasn't been minimized for more than 5 epochs,
1. the **saved weights** were those of the epoch with minimum validation loss (`restore_best_weights=True`).""")

        st.info("""
* **optimizer** = Adam
* **learning_rate** = 0.001
* **loss function** = categorical_crossentropy
* **metrics** = accuracy, custom recall for each class (_e.g._: `Recall(class_id=1, name="recall_COV")` for COVID class)
""")

        st.image("./streamlit_imgs/FittingCustomCNN.png")
        st.markdown("""<center><small><i>On this run, the fitting stopped at 14 epochs, and the saved weights were those of epoch 9</i></small></center>""",
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.beta_expander(label="Show results"):
            st.subheader("`classification_report()`")
            st.image("./streamlit_imgs/results_CNN.png")

            st.subheader("`confusion_matrix()`")
            st.image("./streamlit_imgs/cnfMatrix_customCNN.png")

            st.subheader("`model.evaluate()`")
            st.image("./streamlit_imgs/RobustnessCustomCNN.png", width=350)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""<small>1: Densely Connected Convolutional Networks, Huang _et al._, 2018, [arXiv](https://arxiv.org/pdf/1608.06993.pdf)    
2: [Understanding and visualizing DenseNets](https://towardsdatascience.com/understanding-and-visualizing-densenets-7f688092391a), Pablo Ruiz, 2018</small>""",
                    unsafe_allow_html=True)

    if header_select == "2. Benchmark":
        st.header("2. Benchmark")
        st.image("./streamlit_imgs/results-3class.png")
        st.markdown("""<center><small><i>From Chowdhury</i> et al.<i>, 2020,</i> IEEE Access</small></center>""",
                    unsafe_allow_html=True)

    if header_select == "3. Transfer learning: DenseNet201":

        st.header("3. Transfer learning: DenseNet201")

        dn201 = """We use the DenseNet201 model<sup>1, 2</sup> with its weights set to the **[_ImageNet_](http://www.image-net.org/) dataset** and **without the fully connected layers**
(`include_top = False`) that we will add ourselves."""

        st.markdown(dn201, unsafe_allow_html=True)

        st.subheader("General architecture")

        st.write("Here is represented the architecture of another similar DenseNet variant: DenseNet-121.")

        st.image("./streamlit_imgs/DenseNet121.png")
        st.markdown("""<small><b><u>Architecture of DenseNet-121.</u> Dx:</b> Dense Block x. <b>Tx:</b> Transition Block x. <b>DLx:</b> Dense Layer x.<br>
            <i>From <a href=https://towardsdatascience.com/understanding-and-visualizing-densenets-7f688092391a>Understanding and visualizing DenseNets</a>, by Pablo Ruiz.</i></small>""",
                    unsafe_allow_html=True)

        st.subheader("Last layers")

        st.image("./streamlit_imgs/DN201_archi_cutEnd.png")
        st.markdown("""<small><b><u>Last convolutional block of DenseNet-201.</u></b> <b>Upper row</b> is DenseNet-201's last convolutional block (with weights unfreezed), <b>lower row</b> is our fully connected classifier.</small>""",
                    unsafe_allow_html=True)

        st.subheader("Training hyperparameters")
        st.write("""We did several runs, and found that in addition to unfreezing the dense layers, **unfreezing the last convolutional block of DenseNet201** gave slightly better results.
This could be due to the fact that although DenseNet is well trained on various images (via the ImageNet weights we are using),
there are some useful features specific to our dataset that it can find by fine tuning this last convolutional block.""")

        st.write("""We optimized the model using by minimizing the loss on the validation dataset. The model was fitted over **50 epochs** and:

1. the **learning rate was reduced** by a factor 10 when the loss of the validation set reached a plateau for 3 epochs,
1. the **fitting stopped** when the validation set loss hasn't been minimized for more than 10 epochs,
1. the **model was saved** when validation loss was minimized.""")

        st.info("""
* **optimizer** = Adam
* **learning_rate** = 0.001
* **loss function** = categorical_crossentropy
* **metrics** = accuracy, custom recall for each class (_e.g._: `Recall(class_id=1, name="recall_COV")` for COVID class)
""")

        st.image("./streamlit_imgs/FittingDN201.png")
        st.markdown(
            """<center><small><i>On this run, the fitting stopped at 15 epochs, and the saved weights were those of epoch 5</i></small></center>""",
            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.beta_expander(label="Show results"):
            st.subheader("`classification_report()`")
            st.image("./streamlit_imgs/results_DN201.png")

            st.subheader("`confusion_matrix()`")
            st.image("./streamlit_imgs/cnfMatrix_Iteration2--Finale--DN201--Unfreezed.png")

            st.subheader("`model.evaluate()`")
            st.image("./streamlit_imgs/RobustnessDN201.png", width=350)


    if header_select == "4. Recap":

        st.header("4. Recap")

        st.image("./streamlit_imgs/results_full.png")

        st.subheader("Custom CNN")
        st.image("./streamlit_imgs/cnfMatrix_customCNN.png")

        st.subheader("DenseNet201")
        st.image("./streamlit_imgs/cnfMatrix_Iteration2--Finale--DN201--Unfreezed.png")





# Modelization
############################################################################################################# Prediction demo

from tempfile import NamedTemporaryFile
from predict import predict_img, random_pick

if section == "Prediction demo":

    st.title("Prediction demo")

    st.write("Our DenseNet201 full model was saved _via_ a `ModelCheckpoint` callback during the fitting process. "
             "It can now be loaded in this web app and tested to live predict images!")

    st.header("Predicting on a random image from the testing dataset")

    imageselect = st.selectbox("Pick a random image from the testing dataset:",
                               ["<None>", "NORMAL", "COVID", "Viral Pneumonia"])

    if imageselect != "<None>":

        random_img = random_pick(imageselect)
        st.image(random_img, width=256)

        if st.button("Predict!"):

            gif_runner = st.image(
                "https://aws1.discourse-cdn.com/business7/uploads/streamlit/original/2X/2/247a8220ebe0d7e99dbbd31a2c227dde7767fbe1.gif",
                width=100)

            prediction = predict_img(random_img)

            gif_runner.empty()

            if prediction == "Normal":
                diag = "The image has been predicted as **" + str(prediction) + "**."
                st.success(diag)

            if prediction == "Viral pneumonia":
                diag = "The image has been predicted as **" + str(prediction) + "**."
                st.warning(diag)

            if prediction == "COVID-19+":
                diag = "The image has been predicted as **" + str(prediction) + "**."
                st.error(diag)




    st.header("Predicting on a novel chest X-ray")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    temp_file = NamedTemporaryFile(delete=False)

    if uploaded_file is not None:
        temp_file.write(uploaded_file.getvalue())
        uploaded_img = temp_file.name

        st.image(uploaded_file, caption='Uploaded Image', width=256)

        if st.button("Predict the upload!"):

            gif_runner = st.image("https://aws1.discourse-cdn.com/business7/uploads/streamlit/original/2X/2/247a8220ebe0d7e99dbbd31a2c227dde7767fbe1.gif", width=100)

            prediction = predict_img(uploaded_img)

            gif_runner.empty()

            if prediction == "Normal":
                diag = "The image has been predicted as **" + str(prediction) + "**."
                st.success(diag)

            if prediction == "Viral pneumonia":
                diag = "The image has been predicted as **" + str(prediction) + "**."
                st.warning(diag)

            if prediction == "COVID-19+":
                diag = "The image has been predicted as **" + str(prediction) + "**."
                st.error(diag)



# Prediction demo
############################################################################################################# Improving & understanding the model

if section == "Improving & understanding the model":
    st.title("Improving & understanding the model")

    st.header("Image segmentation")

    st.write("""Our model is imperfect as images are still being misclassified.
One way of improving it would be to **isolate the lungs _via_ image segmentation** in order to eliminate non-relevant image artefacts,
and run our classification model on those segmented images.

Image segmentation is usually a supervised learning task, with an annotated dataset containing annotations of the boundaries
of relevant areas done by a radiologist. This is obviously not the case here, so we tried a **manual unsupervised approach**.""")


    with st.beta_expander("Display the segmentation process"):
        st.image("./streamlit_imgs/ImageSegmentation.png")

    st.markdown("<br>", unsafe_allow_html=True)

    st.header("Feature maps")

    st.write("""Feature maps correspond to **characteristic patterns** in the picture
    and result from the **convolutional product** of the filter (convolution kernel) with the input image.""")

    st.write("_Example of Taj Mahal picture :_")

    st.image("./streamlit_imgs/taj_mahal_conv_process.gif")

    st.write("""In our strategy of improving used models, visualising features 
    maps can help us gain insight into the inner workings of the model and get some 
    understanding of which features the CNN model will detect.""")

    st.image("./streamlit_imgs/Chowdhury2020_fig8.png")
    st.markdown("""<small><i>From Chowdhury</i> et al.<i>, 2020,</i> IEEE Access</small>""",
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.beta_expander("Display input image"):
        st.image("./streamlit_imgs/segm_orig.png")
        st.markdown(
            """<small>Image size is **224 x 224 pixels**</small>""",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.beta_columns(5)

    if col1.button("Convolution 1"):
        st.image("./streamlit_imgs/segm_1.png")


    if col2.button("Convolution 2"):
        st.image("./streamlit_imgs/segm_2.png")


    if col3.button("Convolution 3"):
        st.image("./streamlit_imgs/segm_3.png")


    if col4.button("Convolution 4"):
        st.image("./streamlit_imgs/segm_4.png")


    if col5.button("Convolution 5"):
        st.image("./streamlit_imgs/segm_5.png")



# Improving & understanding the model
############################################################################################################# Conclusion & Perspectives

if section == "Conclusion & Perspectives":
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.title("Conclusion & Perspectives")

    st.header("Conclusions")

    st.write("""The DenseNet201 hybrid model we created via transfer learning gave us really good results, even if we did not do better than the original authors’ benchmark.
Still, this project was a great learning experience that helped us grasp some Deep Learning concepts that were still a bit too abstract for us.
""")

    st.header("Perspectives")

    st.write("""Our project could definitely have benefitted from a bit more time to go more in-depth in several areas, especially in the image segmentation part.
The feature maps are interesting but don’t give us at this moment any insight on specific features that were important for the classification process.
We would have liked to be able to research this subject more.

And of course maybe find novel networks that we could have imported and tested,
instead of using only the state-of-the-art ones available in Keras/TensorFlow.""")

    st.header("Application")

    st.write("""All in all, this model (or a better & stronger version of it :) ) could definitely be implemented in the not-so-distant future, to assist diagnosing patients.
Especially considering the fact that global pandemics such as the current one are known to be a cyclic phenomenon bound to repeat itself every 5-10 years.""")
