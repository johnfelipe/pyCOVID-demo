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