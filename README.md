# modulation-recognition-for-wireless-signals
Algorithm for classification of Digital Modulation such as FSK, PSK, ASK, QAM ........

# For the input dataset used is explained here
The folder " Dataset_input" contains the different input datasets that can be used created by DeepSig.
RadioML 2016.10A : A synthetic dataset, generated with GNU Radio, consisting of 11 modulations (8 digital and 3 analog) at varying SNR levels.
RML2016.10B : Larger Version of RadioML 2016.10a (including AM-SSB).
RadioML 2016.04C : A synthetic dataset, generated with GNU Radio, consisting of 11 modulations. This is a variable-SNR dataset with moderate LO drift, light fading, and numerous different labeled SNR increments for use in measuring performance across different signal and noise power scenarios.

The original datasets can be found Here :
https://www.deepsig.io/datasets

For the data it was generated in a way that captures various channel imperfections that are present in real system using GNU.
Concerning the input datasets that will be used in our first experiment is RadioML 2016.10b, consisting of 160 000 samples, each input is a 2*128 vector, separating real and imaginary parts in complex time samples and labels include SNR ground Truth as well as the modulation type. SNR is between -20db and +18db.
