$ErrorActionPreference = "Stop"

Write-Host "Checking for Python..."
try {
    py --version
}
catch {
    Write-Error "Python launcher (py) is not found. Please install Python."
    exit 1
}

Write-Host "Installing dependencies..."
try {
    py -m pip install -r requirements.txt
}
catch {
    Write-Warning "Failed to install dependencies via pip."
}

Write-Host "Running Border Crossing Traffic Forecasting Pipeline..."
$env:PYTHONPATH = "$PWD"
py src/main.py
