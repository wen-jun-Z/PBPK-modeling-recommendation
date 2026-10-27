[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$CliPath,

    [Parameter(Mandatory = $true)]
    [string]$RuntimeArchivePath,

    [Parameter(Mandatory = $true)]
    [string]$SnapshotPath,

    [Parameter(Mandatory = $true)]
    [string]$ProtocolPath,

    [Parameter(Mandatory = $true)]
    [string]$WorkRoot
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Assert-FileSha256 {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Expected
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file is missing: $Path"
    }
    $Actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne $Expected.ToLowerInvariant()) {
        throw "SHA-256 mismatch for $Path. Expected $Expected, found $Actual."
    }
}

function Get-RelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Path
    )
    $RootUri = [Uri]::new([IO.Path]::GetFullPath($Root).TrimEnd("\") + "\")
    $FileUri = [Uri]::new([IO.Path]::GetFullPath($Path))
    return [Uri]::UnescapeDataString(
        $RootUri.MakeRelativeUri($FileUri).ToString()
    ).Replace("/", "\")
}

function Invoke-CapturedNative {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$StdoutPath,
        [Parameter(Mandatory = $true)][string]$StderrPath
    )
    $Started = [DateTime]::UtcNow
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Executable @Arguments 1> $StdoutPath 2> $StderrPath
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    $Completed = [DateTime]::UtcNow
    return [PSCustomObject]@{
        ExitCode = $ExitCode
        DurationSeconds = ($Completed - $Started).TotalSeconds
    }
}

function Get-ArtifactRows {
    param(
        [Parameter(Mandatory = $true)][string]$SimulationRoot,
        [Parameter(Mandatory = $true)][string]$ResultRoot
    )
    $Rows = @()
    $Files = @(
        Get-ChildItem -LiteralPath $SimulationRoot -Recurse -File |
            Where-Object { $_.Extension.ToLowerInvariant() -in @(".pkml", ".xml", ".csv") } |
            Sort-Object FullName
    )
    foreach ($File in $Files) {
        $Rows += [PSCustomObject]@{
            relative_path = Get-RelativePath -Root $ResultRoot -Path $File.FullName
            simulation_relative_path = Get-RelativePath -Root $SimulationRoot -Path $File.FullName
            byte_size = $File.Length
            sha256 = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            extension = $File.Extension.ToLowerInvariant()
        }
    }
    return $Rows
}

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw "This execution protocol must run on Windows."
}

$CliPath = (Resolve-Path -LiteralPath $CliPath).Path
$RuntimeArchivePath = (Resolve-Path -LiteralPath $RuntimeArchivePath).Path
$SnapshotPath = (Resolve-Path -LiteralPath $SnapshotPath).Path
$ProtocolPath = (Resolve-Path -LiteralPath $ProtocolPath).Path
$WorkRoot = [IO.Path]::GetFullPath($WorkRoot)
if (Test-Path -LiteralPath $WorkRoot) {
    throw "Immutable work root already exists: $WorkRoot"
}

$Protocol = Get-Content -LiteralPath $ProtocolPath -Raw -Encoding UTF8 | ConvertFrom-Json
$ProtocolSha256 = (Get-FileHash -LiteralPath $ProtocolPath -Algorithm SHA256).Hash.ToLowerInvariant()
$WorkerScriptSha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
Assert-FileSha256 -Path $RuntimeArchivePath -Expected $Protocol.runtime.pksim_portable_sha256
Assert-FileSha256 -Path $SnapshotPath -Expected $Protocol.source.snapshot_sha256

$InputDirectory = Join-Path $WorkRoot "input"
$ProjectDirectory = Join-Path $WorkRoot "project"
$RoundTripDirectory = Join-Path $WorkRoot "roundtrip"
$RunRoot = Join-Path $WorkRoot "runs"
$LogDirectory = Join-Path $WorkRoot "logs"
New-Item -ItemType Directory -Path @(
    $InputDirectory,
    $ProjectDirectory,
    $RoundTripDirectory,
    $RunRoot,
    $LogDirectory
) | Out-Null

$LockedWorkerScript = Join-Path $InputDirectory "run_licensed_snapshot_reuse.ps1"
Copy-Item -LiteralPath $PSCommandPath -Destination $LockedWorkerScript
Assert-FileSha256 -Path $LockedWorkerScript -Expected $WorkerScriptSha256

$LockedSnapshot = Join-Path $InputDirectory ([IO.Path]::GetFileName($Protocol.source.snapshot_relative_path))
Copy-Item -LiteralPath $SnapshotPath -Destination $LockedSnapshot
Assert-FileSha256 -Path $LockedSnapshot -Expected $Protocol.source.snapshot_sha256

$VersionStdout = Join-Path $LogDirectory "cli_version.stdout.log"
$VersionStderr = Join-Path $LogDirectory "cli_version.stderr.log"
$VersionResult = Invoke-CapturedNative -Executable $CliPath -Arguments @("--version") `
    -StdoutPath $VersionStdout -StderrPath $VersionStderr
$VersionOutput = (
    (Get-Content -LiteralPath $VersionStdout -Raw -ErrorAction SilentlyContinue) + "`n" +
    (Get-Content -LiteralPath $VersionStderr -Raw -ErrorAction SilentlyContinue)
).Trim()
if ($VersionResult.ExitCode -notin @(0, 1)) {
    throw "PKSim.CLI --version failed with exit code $($VersionResult.ExitCode)."
}
$ExpectedVersion = "PKSim.CLI $($Protocol.runtime.pksim_version)"
if ($VersionOutput -notmatch [Regex]::Escape($ExpectedVersion)) {
    throw "Unexpected PKSim.CLI version: $VersionOutput"
}

$SnapToProjectStdout = Join-Path $LogDirectory "snapshot_to_project.stdout.log"
$SnapToProjectStderr = Join-Path $LogDirectory "snapshot_to_project.stderr.log"
$SnapToProject = Invoke-CapturedNative -Executable $CliPath -Arguments @(
    "snap", "-i", $InputDirectory, "-o", $ProjectDirectory, "-p",
    "-l", (Join-Path $LogDirectory "snapshot_to_project.pksim.log"),
    "--logLevel", "Debug"
) -StdoutPath $SnapToProjectStdout -StderrPath $SnapToProjectStderr
if ($SnapToProject.ExitCode -ne 0) {
    throw "Snapshot reconstruction failed with exit code $($SnapToProject.ExitCode)."
}
$ProjectFiles = @(Get-ChildItem -LiteralPath $ProjectDirectory -Filter "*.pksim5" -File)
if ($ProjectFiles.Count -ne 1 -or $ProjectFiles[0].Length -le 0) {
    throw "Expected exactly one non-empty reconstructed PK-Sim project."
}
$ProjectPath = $ProjectFiles[0].FullName

$RoundTripStdout = Join-Path $LogDirectory "project_to_snapshot.stdout.log"
$RoundTripStderr = Join-Path $LogDirectory "project_to_snapshot.stderr.log"
$RoundTripResult = Invoke-CapturedNative -Executable $CliPath -Arguments @(
    "snap", "-i", $ProjectDirectory, "-o", $RoundTripDirectory, "-s",
    "-l", (Join-Path $LogDirectory "project_to_snapshot.pksim.log"),
    "--logLevel", "Debug"
) -StdoutPath $RoundTripStdout -StderrPath $RoundTripStderr
if ($RoundTripResult.ExitCode -ne 0) {
    throw "Round-trip snapshot export failed with exit code $($RoundTripResult.ExitCode)."
}
$RoundTripFiles = @(Get-ChildItem -LiteralPath $RoundTripDirectory -Filter "*.json" -File)
if ($RoundTripFiles.Count -ne 1 -or $RoundTripFiles[0].Length -le 0) {
    throw "Expected exactly one non-empty round-trip snapshot."
}
$RoundTripPath = $RoundTripFiles[0].FullName

$RunManifests = @()
foreach ($RunId in @("run_1", "run_2")) {
    $CurrentRunRoot = Join-Path $RunRoot $RunId
    New-Item -ItemType Directory -Path $CurrentRunRoot | Out-Null
    $SimulationManifests = @()
    $RunPassed = $true
    foreach ($Simulation in @($Protocol.simulations | Sort-Object sequence)) {
        $SequenceName = "s{0:d4}" -f [int]$Simulation.sequence
        $SimulationRoot = Join-Path $CurrentRunRoot $SequenceName
        New-Item -ItemType Directory -Path $SimulationRoot | Out-Null
        $StdoutPath = Join-Path $SimulationRoot "stdout.log"
        $StderrPath = Join-Path $SimulationRoot "stderr.log"
        $Arguments = @(
            "export", "-p", $ProjectPath, "-o", $SimulationRoot,
            "-s", [string]$Simulation.name,
            "-k", "-x", "-c", "-r",
            "-l", (Join-Path $SimulationRoot "pksim.log"),
            "--logLevel", "Debug"
        )
        $Execution = Invoke-CapturedNative -Executable $CliPath -Arguments $Arguments `
            -StdoutPath $StdoutPath -StderrPath $StderrPath
        $Artifacts = @(Get-ArtifactRows -SimulationRoot $SimulationRoot -ResultRoot $WorkRoot)
        $Extensions = @($Artifacts | ForEach-Object { $_.extension } | Sort-Object -Unique)
        $ReasonCodes = @()
        if ($Execution.ExitCode -ne 0) {
            $ReasonCodes += "PKSIM_CLI_NONZERO_EXIT"
        }
        foreach ($RequiredExtension in @($Protocol.runtime.required_export_extensions)) {
            if ($RequiredExtension -notin $Extensions) {
                $ReasonCodes += "MISSING_REQUIRED_EXPORT_$($RequiredExtension.TrimStart('.').ToUpperInvariant())"
            }
        }
        $Status = if ($ReasonCodes.Count -eq 0) { "passed" } else { "failed" }
        if ($Status -eq "failed") {
            $RunPassed = $false
        }
        $SimulationManifests += [PSCustomObject]@{
            sequence = [int]$Simulation.sequence
            name = [string]$Simulation.name
            status = $Status
            return_code = [int]$Execution.ExitCode
            duration_seconds = [double]$Execution.DurationSeconds
            command = @($CliPath) + $Arguments
            stdout_relative_path = Get-RelativePath -Root $WorkRoot -Path $StdoutPath
            stderr_relative_path = Get-RelativePath -Root $WorkRoot -Path $StderrPath
            artifacts = $Artifacts
            reason_codes = $ReasonCodes
        }
    }
    $RunManifests += [PSCustomObject]@{
        run_id = $RunId
        status = if ($RunPassed) { "passed" } else { "failed" }
        simulations = $SimulationManifests
    }
}

$AllPassed = @($RunManifests | Where-Object { $_.status -ne "passed" }).Count -eq 0
$Manifest = [PSCustomObject]@{
    manifest_schema_version = "1.1.0"
    protocol_id = [string]$Protocol.protocol_id
    protocol_sha256 = $ProtocolSha256
    protocol_text_encoding = "utf-8"
    worker_script_relative_path = Get-RelativePath -Root $WorkRoot -Path $LockedWorkerScript
    worker_script_sha256 = $WorkerScriptSha256
    status = if ($AllPassed) { "passed" } else { "failed" }
    cli_version_output = $VersionOutput
    cli_sha256 = (Get-FileHash -LiteralPath $CliPath -Algorithm SHA256).Hash.ToLowerInvariant()
    runtime_archive_sha256 = (Get-FileHash -LiteralPath $RuntimeArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    input_snapshot_relative_path = Get-RelativePath -Root $WorkRoot -Path $LockedSnapshot
    input_snapshot_sha256 = (Get-FileHash -LiteralPath $LockedSnapshot -Algorithm SHA256).Hash.ToLowerInvariant()
    reconstructed_project_relative_path = Get-RelativePath -Root $WorkRoot -Path $ProjectPath
    reconstructed_project_sha256 = (Get-FileHash -LiteralPath $ProjectPath -Algorithm SHA256).Hash.ToLowerInvariant()
    roundtrip_snapshot_relative_path = Get-RelativePath -Root $WorkRoot -Path $RoundTripPath
    roundtrip_snapshot_sha256 = (Get-FileHash -LiteralPath $RoundTripPath -Algorithm SHA256).Hash.ToLowerInvariant()
    runs = $RunManifests
    completed_at_utc = [DateTime]::UtcNow.ToString("o")
}
$ManifestPath = Join-Path $WorkRoot "windows_run_manifest.json"
$Manifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $ManifestPath -Encoding UTF8

$ChecksumRows = Get-ChildItem -LiteralPath $WorkRoot -Recurse -File |
    Where-Object { $_.Name -ne "files.sha256.csv" } |
    Sort-Object FullName |
    ForEach-Object {
        [PSCustomObject]@{
            RelativePath = Get-RelativePath -Root $WorkRoot -Path $_.FullName
            ByteSize = $_.Length
            SHA256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
$ChecksumRows | Export-Csv -LiteralPath (Join-Path $WorkRoot "files.sha256.csv") `
    -NoTypeInformation -Encoding UTF8

Write-Host "Snapshot reuse execution completed."
Write-Host "Status: $($Manifest.status)"
Write-Host "Result root: $WorkRoot"
if (-not $AllPassed) {
    exit 1
}
