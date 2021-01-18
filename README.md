# HFM_PyTorch
An alternate version of HFM code with PyTorch. 
Right now only utilities.py and Cylinder2D_flower_systematic.py are presented.

- Original Repository is [Hidden Fluid Mechanics](https://github.com/maziarraissi/HFM).
- Original [Paper](https://science.sciencemag.org/content/367/6481/1026.abstract)
- Original [Data and Figures](https://bit.ly/2NRB65U)

## Additional files
- DataManager.py: to save training loss and errors during training.
- test.py: to test the results. 
- plot.py: to plot the results.

## Notes: 
The results are in Results folder. Comparing to the original code, this version is less accurate. 
However, further training can be done by using a smaller learning rate to achieve better error rate. 
An learning rate of 1e-3 (used in the original code) is prone to overfitting and the results are bad. 
An learning rate of 1e-4 (showed in results)can have similar results comparing to the original paper.

## To train:
- download data and place it into a Data folder.

```python Cylinder2D_flower_systematic.py 201 15000 [cuda-device-num|optional] [using visdom|optional]```

```[cuda-device-num]```: don't need if not using GPU
```[using visdom]``` : this was not tested, but was planned to use visdom to visualize training process.
## To test:
- You should have related files in Results folder.
```python test.py v10 [cuda-device-num|optional]```

## To plot: 
- You should have related files in Results folder.
```python plot.py v10```
