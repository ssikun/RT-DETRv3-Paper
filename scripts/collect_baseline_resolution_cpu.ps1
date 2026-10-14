param(
    [string]$Server = 'root@sc01-ssh.gpuhome.cc',
    [int]$Port = 30712,
    [string]$Key = "$env:USERPROFILE\.ssh\codex_rtdetrv3_20261010",
    [string]$BackupDir = 'D:\experiment\paper_backups\baseline40_resolution_cpu_20261014'
)

$ErrorActionPreference = 'Stop'
$ssh = (Get-Command ssh.exe).Source
$scp = (Get-Command scp.exe).Source
$serverRepo = '/root/rivermind-data/RT-DETRv3-Paper'
$serverRun = "$serverRepo/results/baseline40_resolution_cpu_20261014"
New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
$logPath = Join-Path $BackupDir 'collection.log'
$statusPath = Join-Path $BackupDir 'collection.status.json'

try {
    $deadline = (Get-Date).AddHours(14)
    $finished = $false
    while ((Get-Date) -lt $deadline) {
        try {
            $reply = & $ssh -i $Key -p $Port -o BatchMode=yes -o ConnectTimeout=8 $Server `
                "if [ -f '$serverRun/run.status' ]; then cat '$serverRun/run.status'; else printf running; fi" 2>&1 | Out-String
            if ($LASTEXITCODE -eq 0 -and $reply -match 'exit_code=(\d+)') {
                $remoteExit = [int]$Matches[1]
                $finished = $true
                break
            }
        } catch {
            Add-Content -LiteralPath $logPath -Value ("{0:o}: {1}" -f (Get-Date), $_.Exception.Message)
        }
        Start-Sleep -Seconds 30
    }
    if (-not $finished) { throw 'Result collection timed out before a terminal status appeared.' }

    $resultDir = Join-Path $BackupDir 'results'
    New-Item -ItemType Directory -Path $resultDir -Force | Out-Null
    & $scp -i $Key -P $Port -o BatchMode=yes -r "${Server}:$serverRun/." $resultDir
    if ($LASTEXITCODE -ne 0) { throw 'Result transfer failed.' }
    & $scp -i $Key -P $Port -o BatchMode=yes `
        "${Server}:$serverRepo/logs/baseline40_resolution_cpu_20261014_nohup.log" `
        "${Server}:$serverRepo/dataset/visdrone2019/annotations/val.json" $BackupDir
    if ($LASTEXITCODE -ne 0) { throw 'Log/annotation transfer failed.' }

    $baselineSource = 'D:\experiment\paper_backups\eval_weights_clean\baseline40_model_final.pdparams'
    $baselineTarget = Join-Path $BackupDir 'baseline40_model_final.pdparams'
    Copy-Item -LiteralPath $baselineSource -Destination $baselineTarget
    $protocol = Get-Content -LiteralPath (Join-Path $resultDir 'eval_640\protocol.json') -Raw | ConvertFrom-Json
    $baselineHash = (Get-FileHash -LiteralPath $baselineTarget -Algorithm SHA256).Hash.ToLowerInvariant()
    $annotationHash = (Get-FileHash -LiteralPath (Join-Path $BackupDir 'val.json') -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($baselineHash -ne $protocol.weights_sha256 -or $annotationHash -ne $protocol.annotations_sha256) {
        throw 'Baseline or annotation checksum mismatch.'
    }

    $remoteManifest = & $ssh -i $Key -p $Port -o BatchMode=yes -o ConnectTimeout=8 $Server `
        "cd '$serverRun' && find . -type f -exec sha256sum {} +" | Out-String
    if ($LASTEXITCODE -ne 0) { throw 'Remote result checksums unavailable.' }
    $remoteManifest | Set-Content -LiteralPath (Join-Path $BackupDir 'RESULT_SHA256SUMS')
    foreach ($line in ($remoteManifest -split "`r?`n")) {
        if ($line -match '^([0-9a-f]{64})  (.+)$') {
            $expected = $Matches[1]
            $relative = $Matches[2].TrimStart('.', '/')
            $path = Join-Path $resultDir $relative
            $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actual -ne $expected) { throw "Result checksum mismatch: $relative" }
        }
    }
    $summary = @{ remote_exit_code = $remoteExit; verified = $true; completed = (Get-Date).ToString('o') }
    if ($remoteExit -eq 0) {
        $baseline = Get-Content -LiteralPath (Join-Path $resultDir 'eval_640\metrics.json') -Raw | ConvertFrom-Json
        $larger = Get-Content -LiteralPath (Join-Path $resultDir 'eval_960\metrics.json') -Raw | ConvertFrom-Json
        $comparison = @()
        foreach ($name in @('AP','AP50','AP75','APS','APM','APL')) {
            $comparison += [pscustomobject]@{
                metric = $name
                baseline640 = [double]$baseline.$name * 100
                baseline960 = [double]$larger.$name * 100
                delta_points = ([double]$larger.$name - [double]$baseline.$name) * 100
            }
        }
        $comparison | Export-Csv -LiteralPath (Join-Path $BackupDir 'resolution_comparison.csv') -NoTypeInformation
    }
    $summary | ConvertTo-Json | Set-Content -LiteralPath $statusPath
} catch {
    @{ verified = $false; error = $_.Exception.Message; completed = (Get-Date).ToString('o') } |
        ConvertTo-Json | Set-Content -LiteralPath $statusPath
    Add-Content -LiteralPath $logPath -Value $_.Exception.Message
    exit 1
}
