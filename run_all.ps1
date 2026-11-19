# Harmony Risk Guard - M1 pipeline (v0.2 data, fixed P0 issues)
# Fixes vs v0.1: never regenerates working data/ by default; -RegenerateData
# backs up the legacy synthetic folder first and writes only to data/synthetic_v01/.
param(
    [switch]$RegenerateData,   # explicit, backed-up regeneration of legacy synthetic demo data
    [switch]$Smoke,            # 20-sample smoke run of the training pipeline
    [switch]$SkipInstall,      # skip venv/pip setup if already done
    [string]$OnlySteps = ''     # comma-separated step names to run (e.g. -OnlySteps predict_demo)
)
$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

# PS 5.1 defaults to an ANSI console codepage: Chinese literals in this file get
# mangled without a BOM, and native UTF-8 stdout gets decoded as GBK. Force UTF-8
# both ways before any Python output is captured.
chcp 65001 | Out-Null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'

if (-not (Test-Path '.venv')) {
    if ($SkipInstall) { throw '.venv missing; remove -SkipInstall to create it' }
    python -m venv .venv
}
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not $SkipInstall) {
    & $python -m pip install --upgrade pip -q -i https://pypi.tuna.tsinghua.edu.cn/simple
    if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }
    & $python -m pip install -r requirements.txt -q -i https://pypi.tuna.tsinghua.edu.cn/simple
    if ($LASTEXITCODE -ne 0) { throw 'dependency install failed' }
}

if ($RegenerateData) {
    $backup = 'data\synthetic_v01_backup_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
    if (Test-Path 'data\synthetic_v01') {
        New-Item -ItemType Directory -Force -Path $backup | Out-Null
        Copy-Item 'data\synthetic_v01\*' $backup -ErrorAction SilentlyContinue
        Write-Host "BACKUP: data\synthetic_v01 -> $backup"
    }
    & $python training\make_dataset.py
    if ($LASTEXITCODE -ne 0) { throw 'data regeneration failed' }
} else {
    Write-Host 'Data regeneration skipped (default). Working v0.2 splits in data\ are static inputs.'
}

$logDir = 'reports\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'

$trainArgs = @()
if ($Smoke) { $trainArgs = @('--smoke') }

$steps = @(
    @('validate', @('training\validate_dataset.py', '--data-dir', 'data')),
    @('baseline_e0', @('training\baseline.py', '--data-dir', 'data')),
    @('train_e1_head', @('training\train.py', '--device', 'cpu', '--mode', 'head', '--epochs', '20',
                         '--early-stop-patience', '5', '--threads', '8') + $trainArgs),
    @('train_e2_last2', @('training\train.py', '--device', 'cpu', '--mode', 'last2', '--epochs', '20',
                          '--early-stop-patience', '5', '--threads', '8') + $trainArgs),
    @('evaluate_e1_test', @('training\evaluate.py', '--model-dir', 'models\finetuned_head', '--tag', 'e1', '--split', 'test')),
    @('evaluate_e2_test', @('training\evaluate.py', '--model-dir', 'models\finetuned_last2', '--tag', 'e2', '--split', 'test'))
)

# Winner = highest best_val_macro_f1 on the VAL split (test is never used for selection).
$e1 = Get-Content 'reports\training_head.json' -Raw | ConvertFrom-Json
$e2 = Get-Content 'reports\training_last2.json' -Raw | ConvertFrom-Json
$winnerDir = if ($e1.best_val_macro_f1 -ge $e2.best_val_macro_f1) { 'models\finetuned_head' } else { 'models\finetuned_last2' }
Write-Host "WINNER by val macro-F1: $winnerDir (E1=$($e1.best_val_macro_f1) E2=$($e2.best_val_macro_f1))"

$steps += @(
    @('export_onnx', @('training\export_onnx.py', '--model-dir', $winnerDir, '--seq-len', '128')),
    @('validate_onnx', @('training\validate_onnx.py', '--model-dir', $winnerDir, '--samples', '30')),
    @('predict_demo', @('training\predict.py', '--model-dir', $winnerDir, '--instruction', '永久删除用户重要资料',
                        '--tool', 'file.delete', '--target', 'user_document', '--scope', 'recursive',
                        '--authorization', 'ambiguous', '--recoverable', 'false', '--backup-available', 'false',
                        '--classification', 'personal'))
)

if ($OnlySteps) {
    $wanted = $OnlySteps.Split(',') | ForEach-Object { $_.Trim() }
    # @() keeps the result an array: the pipeline unrolls a single match into a scalar otherwise.
    $steps = @($steps | Where-Object { $wanted -contains $_[0] })
    if (-not $steps) { throw "no matching steps in: $OnlySteps" }
}

foreach ($step in $steps) {
    $name = $step[0]
    $log = "$logDir\${stamp}_$name.log"
    Write-Host "RUN: $($step[1] -join ' ')"
    # PS 5.1: with $ErrorActionPreference='Stop' a redirected (2>&1) native-stderr
    # line becomes an ErrorRecord that kills the script. Relax EAP around the
    # call, capture the merged output, write a UTF-8 log, and rely on the
    # explicit exit-code check for failure detection.
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $output = & $python @($step[1]) 2>&1
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    $output | ForEach-Object { '{0}' -f $_ } | Out-File -FilePath $log -Encoding utf8
    $output | ForEach-Object { Write-Host ('{0}' -f $_) }
    if ($code -ne 0) { throw "Step failed: $name (exit $code, see $log)" }
}
Write-Host 'Finished. Inspect models/ and reports/'
