[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BundleRoot,

    [Parameter(Mandatory = $true)]
    [string]$WorkRoot,

    [string]$ScratchRoot = "",

    [int]$MinimumFreeGB = 80
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Assert-Sha256 {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Expected
    )
    $Actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne $Expected.ToLowerInvariant()) {
        throw "SHA-256 mismatch for $Path. Expected $Expected, found $Actual."
    }
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable failed with exit code $LASTEXITCODE."
    }
}

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw "This worker must run on Windows."
}
if (-not [Environment]::Is64BitOperatingSystem) {
    throw "A 64-bit Windows operating system is required."
}

$BundleRoot = (Resolve-Path -LiteralPath $BundleRoot).Path
$WorkRoot = [IO.Path]::GetFullPath($WorkRoot)
$Drive = New-Object IO.DriveInfo([IO.Path]::GetPathRoot($WorkRoot))
$RequiredBytes = [int64]$MinimumFreeGB * 1GB
if ($Drive.AvailableFreeSpace -lt $RequiredBytes) {
    throw "At least $MinimumFreeGB GB free is required on $($Drive.Name)."
}

$ManifestPath = Join-Path $BundleRoot "bundle_manifest.json"
$PlanPath = Join-Path $BundleRoot "execution_plan.json"
$PlanSidecarPath = Join-Path $BundleRoot "execution_plan.sha256"
$RuntimeArchive = Join-Path $BundleRoot "runtime/PKSim-Portable.12.3.173.zip"
$Requirements = Join-Path $BundleRoot "requirements-osp-worker.txt"
foreach ($RequiredPath in @(
    $ManifestPath,
    $PlanPath,
    $PlanSidecarPath,
    $RuntimeArchive,
    $Requirements
)) {
    if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
        throw "Required worker bundle file is missing: $RequiredPath"
    }
}

$Manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$RepositoryArchive = Join-Path `
    $BundleRoot $Manifest.repository_archive_relative_path
if (-not (Test-Path -LiteralPath $RepositoryArchive -PathType Leaf)) {
    throw "Required repository archive is missing: $RepositoryArchive"
}
foreach ($File in $Manifest.files) {
    $Path = Join-Path $BundleRoot $File.relative_path
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Bundle manifest file is missing: $($File.relative_path)"
    }
    Assert-Sha256 -Path $Path -Expected $File.sha256
}
if ([string]::IsNullOrWhiteSpace($ScratchRoot)) {
    $ScratchRoot = Join-Path $Drive.Name (
        "PKX\" + $Manifest.job_id.Substring(9, 12)
    )
}
$ScratchRoot = [IO.Path]::GetFullPath($ScratchRoot)

$Python = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $Python) {
    throw "Python 3.11 or newer must be installed and available as python."
}
$VersionText = & $Python.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect Python version."
}
$VersionParts = $VersionText.Trim().Split(".")
if ([int]$VersionParts[0] -lt 3 -or (
    [int]$VersionParts[0] -eq 3 -and [int]$VersionParts[1] -lt 11
)) {
    throw "Python 3.11 or newer is required; found $VersionText."
}

$env:PYTHONPATH = $BundleRoot
New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null
$RepositoryRoot = Join-Path (
    Join-Path $WorkRoot "repository"
) $Manifest.repository_root_name
$RepositoryParent = Split-Path -Parent $RepositoryRoot
$RepositoryExtractionManifest = Join-Path $RepositoryParent "repository_extraction_manifest.json"
New-Item -ItemType Directory -Force -Path $RepositoryParent | Out-Null
$RepositoryEntries = @(
    $Manifest.files |
        Where-Object {
            $_.relative_path -eq $Manifest.repository_archive_relative_path
        }
)
if ($RepositoryEntries.Count -ne 1) {
    throw "Bundle manifest must contain exactly one locked repository archive."
}
Invoke-Checked $Python.Source -m pbpk_kb.model_assets.windows_extract `
    --archive $RepositoryArchive `
    --destination $RepositoryRoot `
    --manifest $RepositoryExtractionManifest `
    --expected-root-name $Manifest.repository_root_name `
    --expected-sha256 $RepositoryEntries[0].sha256

$Venv = Join-Path $WorkRoot "venv"
if (-not (Test-Path -LiteralPath $Venv -PathType Container)) {
    Invoke-Checked $Python.Source -m venv $Venv
}
$VenvPython = Join-Path $Venv "Scripts/python.exe"
Invoke-Checked $VenvPython -m pip install --disable-pip-version-check -r $Requirements

$RuntimeRoot = Join-Path $WorkRoot "runtime"
$OutputRoot = Join-Path $WorkRoot "execution-results"
Invoke-Checked $VenvPython -m pbpk_kb.model_assets.windows_worker `
    --plan $PlanPath `
    --repository $RepositoryRoot `
    --runtime-archive $RuntimeArchive `
    --runtime-root $RuntimeRoot `
    --output-root $OutputRoot `
    --scratch-root $ScratchRoot

Write-Host "OSP execution worker completed."
Write-Host "Result root: $OutputRoot"
Write-Host "Scratch root: $ScratchRoot"
