
**Exploring the application of deep learning methods for polygenic risk score estimation**

**This repository is being updated
This paper uses several methods to make polygenic (or genetic) risk scores estimates from input SNPs.
There is code for:
MLP: This is a multi-layer perceptron which takes in some number of input SNPs and is trained to predict the final PRS

AE: This is an auto-encoder which is trained to take in a reduced number of input SNPs and outputs a prediction of the original set of SNPs for the specific GRS. Once that prediction is made then the MLP can be run to get the GRS out.

LinearPreds: this is a linear prediction on some number of input SNPs trained to predict the final PRS.

















This code is associated with work done in our paper "Exploring the application of deep learning methods for polygenic risk score estimation":
https://pmc.ncbi.nlm.nih.gov/articles/PMC10760287/

If used please cite:
Squires, Steven, Michael N. Weedon, and Richard A. Oram. "Exploring the application of deep learning methods for polygenic risk score estimation." medRxiv (2023).


