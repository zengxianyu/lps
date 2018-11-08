# lps
code for the paper ``learning to promote saliency detectors"

Environment: python 2.7; pytorch '0.5.0a0+54db14e'; two GTX 1080Ti GPU;

## usage
modify the path to prior maps, images, groud truth and then run test.py

## pre-trained model
[download pre-trained model](https://pan.baidu.com/s/1mOMz6pXYsoJPgqE6hQxI1A)

## performance

This version gives slightly different results from the paper. 

### F-measure
  v  |ECSSD | HKU-IS|PASCALS|DUTS-test|THUR|OMRON|
  --- | --- | ---
SRM  |.8924 | .8739 | .7961 | .7591 |.7079|.7223|
+Ours|.9102 | .9032 | .8054 | .7999 |.7299|.7338|


### MAE
 v   |ECSSD | HKU-IS|PASCALS|DUTS-test|THUR|OMRON|
  --- | --- | ---
SRM  |.0542 | .0459 | .0852 |.0633|.0769|.0767|
+Ours|.0416 | .0330 | .0729 |.0536|.0735|.0696|

I'll add the results on other methods later. 

