# BSC IPC

### The goal of the iPC project is to collect, standardize and harmonize existing clinical knowledge and medical data and, with the help of artificial intelligence, create treatment models for patients.

![alt text](https://lh3.googleusercontent.com/JmACpXhaDCnLVHdpI8y4RVtYaSTG-U9kWD4lC_83PyQZs_hF8VBq7z43-lHgg_F_Mah7qcH3VXDLPdZBD0_qCAMsjnf0rhnt4GVstjftoun9DSVJqjSKN7ZaarFL5ewWbq1IfzLT)

### BSC (Barcelona Supercomputing Center) is the largest research center in Spain and one of the largest supercomputers in Europe. The mission of the Life Sciences department is to understand living organisms by means of theoretical and computational methods (molecular modeling, genomics, proteomics).
<p align="center">
<img src="https://biysc.org/sites/default/files/biysc_bsc_logo.jpg.png" width="300">
  </p>

### Abstract
##### Synthetic data generation is emerging as a dominant solution for personalized medicine as it enables to address critical challenges such as yielding the data volumes needed to deliver accurate results and complying with increasingly restrictive privacy regulations, both demanded in paediatric cancer research. Here we introduce an exaplainable VAE for synthetic data generation for medulloblastoma, a childhood brain tumor. Our model can be used to augment and interpolate available data with synthetic instances, which are automatically annotated with confidence scores to assess the reliability of augmented data points and interpolated paths. The model is transparent as it is able to match the learned latent variables with distinct gene expression patterns. We leverage both the synthetic data generation ability and explainability features of our model to study the unknown relationship between G3 and G4 subgroups of medulloblastoma and identify an intermediate subgroup with a specific gene signature.


## Setup
In order to reproduce the results indicated in the paper simply setup an
environment using the provided `environment.yaml` and `conda` and run the experiments
using the provided makefile:

```bash
conda env create --file environment.yaml
source activate ENV_NAME
```
