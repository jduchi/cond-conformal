Setting up the stats315a environment(s) for using Python, pip, etc.

Step 1: Begin the environment

python -m venv condconformal

source condconformal/bin/activate

(Conda version:
conda create --name stats315a
conda activate stats315a
)

Step 2: Install the appropriate packages

pip install numpy pandas scikit-learn matplotlib
pip install torch torchvision torchaudio
pip install cvxpy
pip install mosek

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

(Or maybe
conda install numpy pandas
conda install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
)

Step 3: Check that PyTorch is using the appropriate GPU stuff
(see the website https://developer.apple.com/metal/pytorch/)

import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

Step 4: Stop your environment

deactivate
