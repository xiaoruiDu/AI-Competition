# Merck Future of AI Challenge submission
https://app.ekipa.de/challenge/future-of-ai/about

## Project name: 
IRENA (Invariant Representations Extraction in Neural Architectures)
## Team: 
NeuroTHIx
## Team members: 
Du Xiaorui, Erdem Yavuzhan, Cristian Axenie

## Abstract
In our proposed solution we aim at a combination of Neuroscience principles, Abstract thinking and Prototyping, towards a solution aiming at bringing the efficiency and robustness of biological intelligence to technical systems that solve real-world problems. We start from the proposed Neuroscientific Research Challenges and propose a novel model and system capable of learning invariant representation.

## Core idea
Using cortical maps as neural substrate for distributed representations of sensory streams, our system is able to learn its connectivity (i.e., structure) from the long-term evolution of sensory observations. This process mimics a typical development process where self-construction (connectivity learning), self-organization, and correlation extraction ensure a refined and stable representation and processing substrate. Following these principles, we propose a model based on Self-Organizing Maps (SOM) and Hebbian Learning (HL) as main ingredients for extracting underlying correlations in sensory data, the basis for subsequently extracting invariant representations.

## Quick Start

1-Download the whole files from the link:

https://drive.google.com/open?id=1I6a21i7N86tNrdttCgA19UEJR9vV_SUQ

Test step:

	1-Set train = 0 in config_FGV.txt(config_IG.txt)

	2-Run

	        python3 trainFGV_dense.py
	
	Distribution_of_DxDyV
	![image](https://github.com/xiaoruiDu/AI-Competition/tree/master/images/Distribution_of_DxDyV.jpg)
	Distribution_of_FxFy
	![image](https://github.com/xiaoruiDu/AI-Competition/tree/master/images/Distribution_of_FxFy.jpg)
	HL matrix of FxFy & GV
	![image](https://github.com/xiaoruiDu/AI-Competition/tree/master/images/HL_matrix_FxFy_GV.jpg)
	Final image.
	![image](https://github.com/xiaoruiDu/AI-Competition/tree/master/images/Optical_Flow.jpg)
	Expected optical flow
	![image](https://github.com/xiaoruiDu/AI-Competition/tree/master/images/expected_flow.jpg)
	Recibstrycted optical flow
	![image](https://github.com/xiaoruiDu/AI-Competition/tree/master/images/reconstructed_flow.jpg)
	

	        python3 trainIG.py
	
	Reconstructed Dx image
	![image](https://github.com/xiaoruiDu/AI-Competition/tree/master/images/rebuild_picture_Dx.jpg)
	Expected Dx Image	
	![image](https://github.com/xiaoruiDu/AI-Competition/tree/master/images/HL_matrix_IG.jpg)

	Note: 

	After runing, you will get results in test_results folder


Train step:

	1-Set train = 1 in config_FGV.txt(config_IG.txt)

	2-Run

	        python3 trainFGV_dense.py

       		python3 trainIG.py

	Note: 

	After runing, you will get .pkl files in the current folder.


