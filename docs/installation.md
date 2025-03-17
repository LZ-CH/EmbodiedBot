## Installation

This code is based on DSPNet.

We test our codes under the following environment:

Ubuntu 20.04
NVIDIA Driver: 550.54.15
CUDA 12.4
Python 3.10.14
PyTorch 2.5.1+cu124

- Create an environment.
    ```shell
    conda create -n embot python=3.10 -y
    conda activate embot
    ```
- Install PyTorch:
    ```shell
    conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
    ```

- Install embodiedqa:
    ```shell
    python install.py all 
    ```
Note: The automatic installation script make each step a subprocess and the related messages are only printed when the subprocess is finished or killed. Therefore, it is normal to seemingly hang when installing heavier packages, such as PyTorch3D.

BTW, from our experience, it is easier to encounter problems when installing these package. Feel free to post your questions or suggestions during the installation procedure.