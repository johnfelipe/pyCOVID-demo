# Streamlit demo

This is the Streamlit demo of our pyCOVID project. The presentation of this project was entirely done on this Streamlit app.

To use it: `streamlit run pyCOVID_StreamLit.py`

Before you do so, absolute paths should be modified in line 7 of `predict.py`.


## Conda environment

A working Conda environment for the Streamlit demo can be created with:

```python
conda create --name strmlt python=3.7.9
conda activate strmlt
pip install tensorflow       # tensorflow==2.4.1
conda install pandas numpy matplotlib opencv streamlit
```
