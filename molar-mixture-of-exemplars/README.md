# A Unified Approach to Semi-Supervised Out-of-Distribution Detection


## Installation

Create a new Python 3.9 environment and install the packages in the requirements.txt file. 


## Running the code

### PAWS

To fit a PAWS model to the CIFAR-10 dataset using frozen DINOv2 embeddings, run

	python main.py +run=base.yaml +R=paws-orig +model/dataset=cifar10 +model/networks=dinov2_vits_14_paws_encoder

### SKMPS

To find the optimal set of prototypes, run the SKMPS selection strategy. First calculate the dataset embeddings

	python main.py +run=base.yaml +R=save-representation +model/dataset=cifar10 +model/networks=dinov2_vits_14_paws_encoder

This will output the dataset embeddings and labels to the folder '../output/local/save-representation_0/tensorboard/version_0/predictions'. Run the script for SKMPS using

	python SKMPS.py

This will generate the output CSV file 'labelled_prototypes/CIFAR10_X.csv' which describes the indices and labels of the selected prototypes.

### vMF-SNE pretraining

To do vMF-SNE pretraining for the PAWS encoder, run

	python main.py +run=base.yaml +R=paws-vMF-SNE +model/dataset=cifar10 +model/networks=dinov2_vits_14_paws_encoder

This will output the pretrained checkpoint to '../output/local/paws-vMF-SNE_0/checkpoints' for the next step.

### MoLAR

Fit MoLAR (supervised) using

	python main.py +run=base.yaml +R=molar +model/dataset=cifar10 +model/networks=dinov2_vits_14_load_head

and fit MoLAR-SS (semi-supervised) use

	python main.py +run=base.yaml +R=molar-SS +model/dataset=cifar10 +model/networks=dinov2_vits_14_load_head

Make sure to do SKMPS and vMF-SNE pretraining first, otherwise this will cause an error.

### OOD evaluation

To run an example OOD evaluation, use

	python eval_ood_detection.py

### Debug flag

To quickly test any of the steps, add +debug=debug to the end of the command and a quick run will be executed.

