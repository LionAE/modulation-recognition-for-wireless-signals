# modulation-recognition-for-wireless-signals
Algorithm for classification of Digital Modulation such as FSK, PSK, ASK, QAM ........

# Short introduction about the problem of Modulation Recognition
We begin with modulation; wich simply a process by wich the desired information to be send is encoded inside an other signal that in turn can be transmitted on a physical medium.
Due to the changing RF environment, the recognition of received signals modulations schemes is becoming challenging and traditionnal methods based on manual features extractions by experts show lack of performance due to new channel conditions and different rates of SNR. So to solve this, new approaches are taken into account, the major one is " Deep Neural Networks", this is firstly due to the high capabilities of DNN for features extractions when it comes with dealing with unsupervised learning, mainly we use CNN (Convolutionnal Neural Networks), and for more advanced architecture with better accuracy, we combine CNN with LSTM or we create a residual connection. 
LSTM are types of RNN ( Reccurent Neural Network), mainly used while working with sequences input. For more details about LSTM, one of the best articles that explain this architecture with highlight on the different gates and their functionnalities can be found in the following link : http://colah.github.io/posts/2015-08-Understanding-LSTMs/
It's an amazing blog post.
So, as we explain it before the call for deep learning architectures for solving the modulation recognition problem is due to its power for feature extraction. We've choosed to work on digital modulations schemes, as we know most of them are based on keying; where values of modulated signal are always one of specific set of predetermined values at all times.

# For the input dataset used is explained here
The folder " Dataset_input" contains the different input datasets that can be used created by DeepSig.
RadioML 2016.10A : A synthetic dataset, generated with GNU Radio, consisting of 11 modulations (8 digitals and 3 analogs) at varying SNR levels.
RML2016.10B : Larger Version of RadioML 2016.10a (including AM-SSB).
RadioML 2016.04C : A synthetic dataset, generated with GNU Radio, consisting of 11 modulations. This is a variable-SNR dataset with moderate LO drift, light fading, and numerous different labeled SNR increments for use in measuring performance across different signal and noise power scenarios.

The original datasets can be found Here :
https://www.deepsig.io/datasets

For the data it was generated in a way that captures various channel imperfections that are present in real system using GNU.
Concerning the input datasets that will be used in our first experiment is RadioML 2016.10b, consisting of 160 000 samples, each input is a 2*128 vector, separating real and imaginary parts in complex time samples and labels include SNR ground Truth as well as the modulation type. SNR is between -20db and +18db.
THe folder "input_data" contains the necessary files in order to generate the input datasets. One can start from here and modify the script in order to generate datas in different conditions depending on the studies.

# Architecture of Deep Neural Networks
First, we clarify that in the beginning we will be using as framework "Keras" with Tesnorflow as back end, it's more suitable for fast developement in order to evaluate our designed architectures. 
For architectures, we will focus on 3 Types :
          - CNN (convolutionnal Neural Networks)
          - ResNet (CNN with residual connection)
          - CNN_LSTM (CNN followed by an LSTM network)
          
Details about the architectures, performance and comments can be found in each specific folder for this Architecture.

Better architectures will be developped along the way addressing different issues, so stay Tuned for better work ! 
For practicality, we would love to adapt the best and choosen architecture for researchers who are in the area of communications, by writting it in MATLAB using 'Deep learning toolbox'. But unfortunately, this toolbox is no longer maintained and updated, we will investigate this, and try to figure out if it's possible to use Neural Network Toolbox in matlab for this issue. Furthermore, we project soon to develop an entire designed architecture that solve other challenges in modulation recognition problem using tensorflow with high level API, in the following weeks. 
