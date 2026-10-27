param(
    [Parameter(Mandatory = $true)]
    [string]$CliPath,

    [Parameter(Mandatory = $true)]
    [string]$SnapshotPath,

    [Parameter(Mandatory = $true)]
    [string]$WorkRoot,

    [ValidateSet(
        "caffeine_study_scenarios_v3.json",
        "caffeine_study_scenarios_v4.json",
        "caffeine_study_scenarios_v5.json"
    )]
    [string]$SnapshotFileName = "caffeine_study_scenarios_v5.json"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScenarioNames = @(
    "CAL_BLANCHARD_IV",
    "CAL_BLANCHARD_PO",
    "HOLD_WAHLLANDER_IV",
    "EXT_EUDRACT_PO",
    "EXT_PMC535_PO",
    "EXT2_PMC572_PO_MALE",
    "EXT2_PMC572_PO_FEMALE"
)

function Invoke-PKSimCli {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $CliPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "PKSim.CLI failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Assert-NonEmptyFile {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file is missing: $Path"
    }
    if ((Get-Item -LiteralPath $Path).Length -le 0) {
        throw "Required file is empty: $Path"
    }
}

$CliPath = (Resolve-Path -LiteralPath $CliPath).Path
$SnapshotPath = (Resolve-Path -LiteralPath $SnapshotPath).Path
$WorkRoot = [System.IO.Path]::GetFullPath($WorkRoot)
if (Test-Path -LiteralPath $WorkRoot) {
    throw "Immutable work root already exists: $WorkRoot"
}

$InputDirectory = Join-Path $WorkRoot "input"
$ProjectDirectory = Join-Path $WorkRoot "project"
$RoundTripDirectory = Join-Path $WorkRoot "roundtrip"
$ExportDirectory = Join-Path $WorkRoot "exports"
$LogDirectory = Join-Path $WorkRoot "logs"
New-Item -ItemType Directory -Path @(
    $InputDirectory,
    $ProjectDirectory,
    $RoundTripDirectory,
    $ExportDirectory,
    $LogDirectory
) | Out-Null

$LockedSnapshot = Join-Path $InputDirectory $SnapshotFileName
Copy-Item -LiteralPath $SnapshotPath -Destination $LockedSnapshot
Assert-NonEmptyFile -Path $LockedSnapshot

$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$VersionOutput = & $CliPath --version 2>&1 | Out-String
$VersionExitCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorActionPreference
if ($VersionExitCode -notin @(0, 1)) {
    throw "PKSim.CLI --version failed with exit code $VersionExitCode"
}
if ($VersionOutput -notmatch "PKSim\.CLI 12\.3\.173") {
    throw "Unexpected PKSim.CLI version: $VersionOutput"
}
$VersionOutput | Set-Content -LiteralPath (Join-Path $LogDirectory "cli_version.log")

Invoke-PKSimCli -Arguments @(
    "snap",
    "-i", $InputDirectory,
    "-o", $ProjectDirectory,
    "-p",
    "-l", (Join-Path $LogDirectory "snapshot_to_project.log"),
    "--logLevel", "Debug"
)

$ProjectFiles = @(Get-ChildItem -LiteralPath $ProjectDirectory -Filter "*.pksim5" -File)
if ($ProjectFiles.Count -ne 1) {
    throw "Expected exactly one reconstructed PK-Sim project; found $($ProjectFiles.Count)."
}
$ProjectPath = $ProjectFiles[0].FullName
Assert-NonEmptyFile -Path $ProjectPath

Invoke-PKSimCli -Arguments @(
    "snap",
    "-i", $ProjectDirectory,
    "-o", $RoundTripDirectory,
    "-s",
    "-l", (Join-Path $LogDirectory "project_to_snapshot.log"),
    "--logLevel", "Debug"
)

$RoundTripFiles = @(Get-ChildItem -LiteralPath $RoundTripDirectory -Filter "*.json" -File)
if ($RoundTripFiles.Count -ne 1) {
    throw "Expected exactly one round-trip snapshot; found $($RoundTripFiles.Count)."
}
Assert-NonEmptyFile -Path $RoundTripFiles[0].FullName

foreach ($RunId in @("run_1", "run_2")) {
    $RunDirectory = Join-Path $ExportDirectory $RunId
    New-Item -ItemType Directory -Path $RunDirectory | Out-Null
    $Arguments = @(
        "export",
        "-p", $ProjectPath,
        "-o", $RunDirectory,
        "-s"
    ) + $ScenarioNames + @(
        "-k", "-x", "-c", "-r",
        "-l", (Join-Path $LogDirectory "export_$RunId.log"),
        "--logLevel", "Debug"
    )
    Invoke-PKSimCli -Arguments $Arguments

    foreach ($ScenarioName in $ScenarioNames) {
        $ScenarioDirectory = Join-Path $RunDirectory $ScenarioName
        Assert-NonEmptyFile -Path (
            Join-Path $ScenarioDirectory "$ScenarioName-Results.csv"
        )
        Assert-NonEmptyFile -Path (
            Join-Path $ScenarioDirectory "$ScenarioName.xml"
        )
        Assert-NonEmptyFile -Path (
            Join-Path $ScenarioDirectory "$ScenarioName.pkml"
        )
    }
}

$ManifestRows = Get-ChildItem -LiteralPath $WorkRoot -Recurse -File |
    Where-Object { $_.Name -ne "files.sha256.csv" } |
    Sort-Object FullName |
    ForEach-Object {
        $RootUri = [Uri]::new($WorkRoot.TrimEnd("\") + "\")
        $FileUri = [Uri]::new($_.FullName)
        $RelativePath = [Uri]::UnescapeDataString(
            $RootUri.MakeRelativeUri($FileUri).ToString()
        ).Replace("/", "\")
        [PSCustomObject]@{
            RelativePath = $RelativePath
            ByteSize = $_.Length
            SHA256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
$ManifestRows | Export-Csv -LiteralPath (
    Join-Path $WorkRoot "files.sha256.csv"
) -NoTypeInformation -Encoding UTF8

[PSCustomObject]@{
    Status = "completed"
    WorkRoot = $WorkRoot
    ProjectPath = $ProjectPath
    RoundTripSnapshotPath = $RoundTripFiles[0].FullName
    SnapshotFileName = $SnapshotFileName
    ScenarioCount = $ScenarioNames.Count
    RunCount = 2
    CompletedAtUtc = [DateTime]::UtcNow.ToString("o")
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (
    Join-Path $WorkRoot "windows_run_manifest.json"
) -Encoding UTF8
