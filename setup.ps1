py -3.11 -m venv venv --system-site-package
. venv\Scripts\Activate.ps1

$pytorch_output = python -c "import torch; print(torch.__version__)" 2>&1
if ($pytorch_output -match "No module named 'torch'") {
    Write-Host "PyTorch is not installed." -ForegroundColor Red
    Write-Host "The verified version is 2.4.0+cu124 and install command is pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124" -ForegroundColor Yellow
    exit
} else {
    Write-Host "PyTorch is installed. Version: $pytorch_output" -ForegroundColor Green
}

pip install gymnasium # gymnasium==0.29.1
pip install stable-baselines3[extra] # stable_baselines3[extra]==2.3.2
pip install tensorboard # tensorboard==2.17.0
