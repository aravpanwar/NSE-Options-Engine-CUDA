# PowerShell wrapper for running the engine inside the Docker container.
# Usage:
#   .\run.ps1                    - build all native targets inside container
#   .\run.ps1 build              - same as above
#   .\run.ps1 lsm_engine [args]  - run a binary (e.g. lsm_engine --paths 1000000)
#   .\run.ps1 shell              - drop into an interactive shell in the container

param(
    [Parameter(Position = 0)][string]$Cmd = "build",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)

$ErrorActionPreference = "Stop"
$image = "nse-cuda"

$null = docker image inspect $image 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Image '$image' not found, building (one-time, ~5 min)..."
    docker build -t $image .
    if ($LASTEXITCODE -ne 0) { throw "docker build failed" }
}

$vol = "${PWD}:/app"

switch ($Cmd) {
    "build" { docker run --rm --gpus all -v $vol $image build }
    "shell" { docker run --rm -it --gpus all -v $vol $image bash }
    default { docker run --rm --gpus all -v $vol $image "./bin/$Cmd" @Rest }
}
