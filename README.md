[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/carolinep-ds/pycovid-demo)


# pyCOVID: Deep learning CNN classification of COVID-19+, normal and viral pneumonia chest X-rays

This is the Streamlit demo of our pyCOVID project. The presentation of this project was entirely done
on this interactive [Streamlit app](https://share.streamlit.io/carolinep-ds/pycovid-demo).


It is currently hosted on Streamlit's platform, but you can also run it locally.
Instructions to do so are provided below.


## Conda environment

A working Conda environment for the Streamlit demo can be created with:

```python
conda create --name strmlt python=3.7.9
conda activate strmlt
pip install tensorflow==2.4.1
conda install pandas numpy matplotlib opencv streamlit
```

## How to run this demo

```
streamlit run https://raw.githubusercontent.com/CarolineP-DS/pyCOVID-demo/master/streamlit_app.py
```

## Notebooks

The notebooks containing the code used in this project are located in the `notebooks` folder of this repo.

Several notebooks were handed in at each step of the project, following instructions and discussions with our mentor. The main notebooks of the project were stitched together to form `pyCOVID_Full_notebook`:

1. Exploratory Data Analysis
2. DataViz
3. First iteration of our deep learning model (custom CNN)
4. Second iteration: hybrid model _via_ transfer learning

Supplementary notebooks are provided: they produced the results shown in the Streamlit demo.

All notebooks are best seen on Google Colab with dark theme.


| Notebook                                  	| GoogleColaboratory 	| What it is	|
| -------------------------------------------	|---------------------------	| ---------------------------------------------------------------------------------------------------------------------------------------------------	|
| `pyCOVID_Full_notebook`             	| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CarolineP-DS/pyCOVID-demo/blob/master/notebooks/pyCOVID_Full_notebook.ipynb)	| Stitched the 4 steps and adapted it to Google Colab. Best run for **custom CNN**.                                                                                         	|
| `Iteration2_Finale_DN201_Unfreezed` 	| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CarolineP-DS/pyCOVID-demo/blob/master/notebooks/Iteration2_Finale_DN201_Unfreezed.ipynb)	| Best run for **hybrid DenseNet201** model (with its unfreezed last convolution block and our own classifier), which produced the `model.h5` saved model file used in our demo	|
| `Iteration2_Finale_InceptionV3`     	| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CarolineP-DS/pyCOVID-demo/blob/master/notebooks/Iteration2_Finale_InceptionV3.ipynb) 	| Best run for **InceptionV3** (freezed and with our own classifier)                                                                                                         	|
| `Iteration2_Finale_VGG16`           	| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CarolineP-DS/pyCOVID-demo/blob/master/notebooks/Iteration2_Finale_VGG16.ipynb)	| Best run for **VGG16** (freezed and with our own classifier)                                                                                                              	|
