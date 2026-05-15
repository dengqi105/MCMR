
# 
<div align="center">
<img src="https://github.com/dengqi105/MCMR/blob/main/main.png" width="700" />
</div>

## How to use the code
### Requirements\*

* numpy, pandas, scipy, scikit-learn
* [vmdpy](https://pypi.org/project/vmdpy/) (optional) and [ewtpy](https://pypi.org/project/ewtpy/) (optional)

### Key parameters
It is suitable for vibration, sound, displacement, torque, temperature, and electrical vortex signals.
```
fs:           the sampling frequency

K:            the number of FIMFs, 5~9 is suitable for most situations.

d_method:     AFD has an excellent performance.

corr:         the correlation measure method. The default is the Correntropy.

clf:          the classifier. The default is the Ridge regression.

z_score:      the data normalization method.
```

### Examples

MCMR incorporates intra-sensor scale-varied  and inter-sensor scale-aligned correlation (corresponding to intra_features and inner_features in the code). The number of inner_features and intra_features are (K+1)\*S\*S and S\*(K+1)\*(K+1), respectively. 

Note that the format of input X should be (Examples, Sensors, Length).  fs (sampling frequency) is a must for "AFD", and kernel_size is indispensable for "Correntropy". 

```python
Extractor = MCMR (fs, K,kernel_size=45,d_method="AFD",corr="Correntropy",clf="RR",z_score=True)

features, inner_features, intra_features = Extractor.transform(X)
```

After feature extraction, the result can be predicted using the following code.  X_train and Y_train are the training set and corresponding labels.  X_test is the testing set.

```python

Y_pre = Extractor.fit_predict(X_train, Y_train, X_test)

```
## Results under time-varying conditions
The WT dataset in collaboration with Weite Technologies is not available for open source sharing due to confidentiality and proprietary considerations. The [KAIST](https://data.mendeley.com/datasets/vxkj334rzv/7) [HIT](https://github.com/HouLeiHIT/HIT-dataset) [MCC5-THU](https://data.mendeley.com/datasets/p92gj2732w/2) [SEU](https://github.com/cathysiyu/Mechanical-datasets) [UOEMD](https://data.mendeley.com/datasets/msxs4vj48g/1) dataset can be downloaded directly.

| dataset |     number of sensors(S)     	| sample length(L) 	|K| required training sample number for 95% accuracy|  required training sample number for 99% accuracy|
|:------:	  |:-----------------:|:------------:	| :------------:	| :------------:	| :------------:	|
|    KAIST        |     4   	| 2700 	|9| 15	| 40	|
|    HIT   	      |     6   	| 512 	|7| 6	  | 13	|
|  MCC5-THU(speed)|     6   	| 2456 	|5| 10	| 33	|
|  MCC5-THU(load) |     6   	| 1842 	|5| 2   | 30	|
|  UOEMD(unload)  |     5   	| 2800 	|5| 13  | 35	|
|  UOEMD(load)    |     5   	| 2800 	|7| 25  | 75	|
## Results under constant speed conditions
| dataset 	|     number of sensors(S)     	| sample length(L) 	|K| required training sample number for 95% accuracy|  required training sample number for 99% accuracy|
|:------:	    |:-----------------:	|:------------:	| :------------:	| :------------:	| :------------:	|
|    KAIST    |     4   	| 2700 	|9| 1	| 2	|
| SEU(gearbox)|     8   	| 512 	|7| 3	| 4	|
| SEU(bearing)|     8   	| 512 	|7| 2	| 7	|
|UOEMD(unload)|     5     | 2800 	|5| 2 | 3 |
|UOEMD(load)  |     5     | 2800 	|5| 2 | 3 |

## Acknowledgement
Z. Feng, Q. Wu, S. Yang, Temporal local correntropy representation for fault diagnosis of machines, IEEE Trans. Ind. Inform. 19 (12) (2023) 11868-11877.

Singh, S.D. Joshi, R.K. Patney, K. Saha, The Fourier decomposition method for nonlinear and non-stationary time series analysis, P. Roy. Soc. a-Math. Phy. 473 (2199) (2017).

## Note
The released code is an engineering-oriented implementation of MCMR with practical modifications for stability and efficiency, so some details differ slightly from the original formulation.

