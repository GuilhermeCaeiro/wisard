# WiSARD
The WiSARD model, originally created by Alexander et al. [1], is a Weightless Neural Network that uses RAM-based (and by "RAM", it is meant "Random Access Memory") discriminators to classify patterns. For more information, some available resources (articles) that can be recommended are [2] and [3] (they also briefly describe other weightless models). 

The code present in this repository is just a **TEST** implementation of the WiSARD classifier in Python and might be useful only for educational purposes (the code is very "crude" and unoptimized). It also includes mental image generation, a technique presented by Grieco et al. in [4]. For those wanting a mature and more efficient implementation, it is suggested the [wisardpkg](https://github.com/IAZero/wisardpkg) library (developed in C++, but with an interface for Python).

## Additional Information
This software was developed using Python 3 and should be compatible with all Python 3 subversions. 

To run WiSARD and the provided tests, the following libraries are required:
 - cython
 - numpy
 - matplotlib
 - sklearn
 - pandas
 - 

## References
[1] I. Aleksander,  W. Thomas, P. Bowden, WISARD: a radical new step forward in image recognition, Sensor Review 4(3) (1984) 120-124.

[2] T.B. Ludermir, A. Carvalho, A.P. Braga, M.C.P. Souto, Weightless Neural Models: A Review of Current and Past Works, Neural Computing Surveys 2 (1999) 41-60.

[3] I. Aleksander, M. De Gregorio, F.M.G. França, P.M.V. Lima, H. Morton, A brief introduction to Weightless Neural Systems, European Symposium on Artificial Neural Networks proceedings (2009) 299-305.

[4] B.P.A. Grieco, P.M.V. Lima, M. De Gregorio, F.M.G. França, Producing pattern examples from "mental" images, Neurocomputing 73 (7-9) (2010) 1057-1064.
