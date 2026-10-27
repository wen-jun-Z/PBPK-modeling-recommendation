[CmdletBinding()]
param(
    [string]$BundleRoot = $PSScriptRoot,
    [Parameter(Mandatory = $true)][string]$ResultsRoot
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Assert-Sha256 {
    param([string]$Path, [string]$Expected)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file is missing: $Path"
    }
    $Actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne $Expected.ToLowerInvariant()) {
        throw "SHA-256 mismatch for $Path. Expected $Expected, found $Actual."
    }
}

$BundleRoot = (Resolve-Path -LiteralPath $BundleRoot).Path
$ManifestPath = Join-Path $BundleRoot "batch_manifest.json"
$JobsPath = Join-Path $BundleRoot "jobs.jsonl"
$RunnerPath = Join-Path $BundleRoot "run_licensed_snapshot_reuse.ps1"
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$RunnerSha256 = (Get-FileHash -LiteralPath $RunnerPath -Algorithm SHA256).Hash.ToLowerInvariant()
foreach ($File in @($Manifest.files)) {
    $LockedPath = Join-Path $BundleRoot $File.relative_path
    Assert-Sha256 -Path $LockedPath -Expected $File.sha256
    if ((Get-Item -LiteralPath $LockedPath).Length -ne [int64]$File.byte_size) {
        throw "Byte-size mismatch for $LockedPath."
    }
}
Assert-Sha256 -Path $JobsPath -Expected $Manifest.jobs_sha256
$RuntimeArchive = Join-Path $BundleRoot $Manifest.runtime_relative_path
Assert-Sha256 -Path $RuntimeArchive -Expected $Manifest.runtime_sha256

$RuntimeRoot = Join-Path $ResultsRoot ("runtime-" + $Manifest.runtime_sha256.Substring(0, 16))
$RuntimeMarker = Join-Path $RuntimeRoot "runtime.sha256"
if (-not (Test-Path -LiteralPath $RuntimeMarker -PathType Leaf)) {
    if (Test-Path -LiteralPath $RuntimeRoot) {
        throw "Incomplete runtime directory already exists: $RuntimeRoot"
    }
    New-Item -ItemType Directory -Path $RuntimeRoot -Force | Out-Null
    Expand-Archive -LiteralPath $RuntimeArchive -DestinationPath $RuntimeRoot
    $Manifest.runtime_sha256 | Set-Content -LiteralPath $RuntimeMarker -Encoding ASCII
}
if ((Get-Content -LiteralPath $RuntimeMarker -Raw -Encoding ASCII).Trim() -ne $Manifest.runtime_sha256) {
    throw "Runtime marker checksum does not match the batch manifest."
}
$Cli = @(Get-ChildItem -LiteralPath $RuntimeRoot -Recurse -Filter "PKSim.CLI.exe" -File)
if ($Cli.Count -ne 1) {
    throw "Expected exactly one PKSim.CLI.exe under $RuntimeRoot; found $($Cli.Count)."
}

New-Item -ItemType Directory -Path $ResultsRoot -Force | Out-Null
$Rows = @()
$Jobs = @(
    Get-Content -LiteralPath $JobsPath -Encoding UTF8 |
        Where-Object { $_.Trim() } |
        ForEach-Object { $_ | ConvertFrom-Json }
)
foreach ($Job in $Jobs) {
    $ProtocolPath = Join-Path $BundleRoot $Job.protocol_relative_path
    $SnapshotPath = Join-Path $BundleRoot $Job.snapshot_relative_path
    Assert-Sha256 -Path $ProtocolPath -Expected $Job.protocol_sha256
    Assert-Sha256 -Path $SnapshotPath -Expected $Job.snapshot_sha256
    $WorkRoot = Join-Path $ResultsRoot $Job.protocol_key
    $RunManifest = Join-Path $WorkRoot "windows_run_manifest.json"
    $Status = "not_started"
    $Detail = $null
    if (Test-Path -LiteralPath $RunManifest -PathType Leaf) {
        $Existing = Get-Content -LiteralPath $RunManifest -Raw -Encoding UTF8 | ConvertFrom-Json
        $ExistingWorkerSha256 = if (
            $null -ne $Existing.PSObject.Properties["worker_script_sha256"]
        ) { [string]$Existing.worker_script_sha256 } else { "" }
        if (
            $Existing.protocol_id -ne $Job.protocol_id -or
            $Existing.protocol_sha256 -ne $Job.protocol_sha256 -or
            $ExistingWorkerSha256 -ne $RunnerSha256
        ) {
            $Status = "blocked_existing_result_identity_mismatch"
            $Detail = "Existing result protocol identity, protocol checksum, or worker checksum does not match."
        }
        elseif ($Existing.status -eq "passed") {
            $Status = "already_executed_passed"
        }
        elseif ($Existing.status -eq "failed") {
            $Status = "already_executed_negative"
            $Detail = "Existing checksum-locked execution completed with a negative outcome."
        }
        else {
            $Status = "blocked_existing_result_unknown_status"
            $Detail = "Existing result manifest has an unsupported terminal status: $($Existing.status)"
        }
    }
    elseif (Test-Path -LiteralPath $WorkRoot) {
        $Status = "blocked_incomplete_existing_result"
        $Detail = "Work root exists without windows_run_manifest.json; it was not overwritten."
    }
    else {
        try {
            $RunnerArguments = @(
                "-NoProfile",
                "-ExecutionPolicy", "Bypass",
                "-File", $RunnerPath,
                "-CliPath", $Cli[0].FullName,
                "-RuntimeArchivePath", $RuntimeArchive,
                "-SnapshotPath", $SnapshotPath,
                "-ProtocolPath", $ProtocolPath,
                "-WorkRoot", $WorkRoot
            )
            & powershell.exe @RunnerArguments
            $RunnerExitCode = $LASTEXITCODE
            if (-not (Test-Path -LiteralPath $RunManifest -PathType Leaf)) {
                $Status = "runner_failed"
                $Detail = "Worker exited with code $RunnerExitCode without a result manifest."
            }
            else {
                $Executed = Get-Content -LiteralPath $RunManifest -Raw -Encoding UTF8 | ConvertFrom-Json
                $ExecutedWorkerSha256 = if (
                    $null -ne $Executed.PSObject.Properties["worker_script_sha256"]
                ) { [string]$Executed.worker_script_sha256 } else { "" }
                if (
                    $Executed.protocol_id -ne $Job.protocol_id -or
                    $Executed.protocol_sha256 -ne $Job.protocol_sha256 -or
                    $ExecutedWorkerSha256 -ne $RunnerSha256
                ) {
                    $Status = "blocked_result_identity_mismatch"
                    $Detail = "New result protocol identity, protocol checksum, or worker checksum does not match."
                }
                elseif ($Executed.status -eq "passed") {
                    $Status = "executed_passed"
                    $Detail = "Worker exit code: $RunnerExitCode"
                }
                elseif ($Executed.status -eq "failed") {
                    $Status = "executed_negative"
                    $Detail = "Checksum-locked execution completed with a negative outcome; worker exit code: $RunnerExitCode"
                }
                else {
                    $Status = "runner_failed"
                    $Detail = "Worker produced unsupported result status '$($Executed.status)'; exit code: $RunnerExitCode"
                }
            }
        }
        catch {
            $Status = "runner_failed"
            $Detail = $_.Exception.Message
        }
    }
    $Rows += [PSCustomObject]@{
        protocol_key = [string]$Job.protocol_key
        protocol_id = [string]$Job.protocol_id
        full_name = [string]$Job.full_name
        version_id = [string]$Job.version_id
        status = $Status
        detail = $Detail
        result_root = $WorkRoot
    }
    $Rows | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath `
        (Join-Path $ResultsRoot "batch_progress.json") -Encoding UTF8
}

$Summary = [PSCustomObject]@{
    completed_at_utc = [DateTime]::UtcNow.ToString("o")
    job_count = $Rows.Count
    executed_count = @(
        $Rows | Where-Object { $_.status -in @(
            "executed_passed", "already_executed_passed",
            "executed_negative", "already_executed_negative"
        ) }
    ).Count
    passed_count = @(
        $Rows | Where-Object { $_.status -in @("executed_passed", "already_executed_passed") }
    ).Count
    negative_count = @(
        $Rows | Where-Object { $_.status -in @("executed_negative", "already_executed_negative") }
    ).Count
    runner_failed_count = @($Rows | Where-Object { $_.status -eq "runner_failed" }).Count
    blocked_count = @($Rows | Where-Object { $_.status -like "blocked_*" }).Count
    jobs = $Rows
}
$Summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath `
    (Join-Path $ResultsRoot "batch_summary.json") -Encoding UTF8
if ($Summary.runner_failed_count -gt 0 -or $Summary.blocked_count -gt 0) { exit 1 }
