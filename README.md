# cgmr

Cascaded Gaussian Mixture Regression (C-GMR)
Matlab source codes for training and using the Integrated C-GMR (IC-GMR) model and the J-GMR model
------------------------------------

HOW-to-CITE: 
When referring to IC-GMR, please cite: Hueber, T., Girin, L., Alameda-Pineda, X., Bailly, G. (2015) "Speaker-Adaptive Acoustic-Articulatory Inversion using Cascaded Gaussian Mixture Regression", in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 23, no. 12, pp. 2246-2259 
When referring to the J-GMR, please cite: Girin, L, Hueber, T., Alameda-Pineda, X.,(2017) "Extending the Cascaded Gaussian Mixture Regression Framework for Cross-Speaker Acoustic-Articulatory Mapping", in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 25, no. 3, pp. 662-673

LICENSE: 
C-GMR is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version. 

C-GMR is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
 
CONTENT: 
- cgmmTrain: Training routine for IC-GMR model (given a training set of N {input (x), output (y)} observations and an adaptation set of N0 {input (z)} observations with typically N0 << N)
- cgmmMap: Mapping routine for converting input (z) to output (y) observations using IC-GMR (MSE criterion)
- jgmmTrain:  Training routine for J-GMR model (given a training set of N {input (x), output (y)} observations and an adaptation set of N0 {input (z)} observations with typically N0 << N)
- jgmmMap: Mapping routine for converting input (z) to output (y) observations using J-GMR (MSE criterion)
- gmmCalculateLikelihood & gmmCalculatePosteriors: subfunctions for calculating likelihood and posterior probabilities, adapted from GMR training routine of Dr. Sylvain Calinon (LASA Lab, EPFL). 


CONTACT: Thomas Hueber, thomas.hueber@gipsa-lab.fr
------------
