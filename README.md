# Separable Scale-Equivariant Object Percetion
Source codes for article entitled ["Scale-Equivariant Object Perception for Autonomous Driving"](https://ieeexplore.ieee.org/document/10480255).  
In this repository, we briefly introduce our code and provide additional information to enhance clarity.

## Environment
### Prerequisite
1. mmcv-full == 2.0.1
2. mmdetection >= 3.1.0
    * This repository only includes modified parts of the mmdetection. For complete details, please clone the corresponding version.
3. mmengine
4. KITTI Dataset

### Data
Please run tools/data_converter/kitti_converter.py to generate data pickle file.

### Train & Test
Please run tools/run_train.sh and run_test.sh, for example:
```bash
bash tools/run_train.sh
```
In config files, you can modify permute options. These options control the type of basis functions used:

1. Scale ***combined*** basis functions: Fuse scales in each scale dimension.
2. Scale ***isolated*** basis functions: Use one scale in each scale dimension.

You can set it up for your purpose.

### Erratum and Additional Information
After our paper was accepted in IEEE Transactions on Intelligent Vehicles (IEEE T-IV), we found some errors in the setup details that were not addressed in the initial article.   

* Basis function format:
    * The results presented in the paper utilized scale ***combined*** basis functions. These functions fuse scales in each scale dimension through tensor reorganization for all scale-equivariant basis functions, potentially reducing scale-equivariance errors.
    * However, it's important to note that the original authors of prior scale-equivariant works introduced ***isolated*** basis functions. Our work therefore includes modifications to this approach.

* Additional activation functions:
    * We implemented additional activation functions before the residual connection in all scale-equivariant backbones, with the exception of $x_{LF}$ identity down-sampling layers.
    * This modification also contributes to the reduction of scale-equivariance errors.

**We inadvertently omitted these details from the initial article. To improve readability, we requested that IEEE T-IV add this correction to be incorporated in the final edit.**   
**We apologize for this oversight.**

For transparency, we have included experimental results from both the ***isolated*** and our modified ***combined*** approaches in the 'results' folder.   
These findings are available in the enclosed file for your reference. Our analysis reveals:

* Detection: The isolated basis generally surpassed the combined basis in detection tasks.
* Tracking Metrics: Prior methods using the isolated basis achieved comparable results to our proposed approach.

These comprehensive results offer a balanced view of performance differences, enhancing the clarity and integrity of our research.

### Acknowledgements
We'd like to express our appreciation to the authors who presented nice research [SESN](https://github.com/ISosnovik/sesn) and [DISCO](https://github.com/ISosnovik/disco).   
And we also thank to [MMDetection](https://github.com/open-mmlab/mmdetection) in facilitating our implementations.