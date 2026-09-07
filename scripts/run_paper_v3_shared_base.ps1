$ErrorActionPreference = "Stop"
$sggtPython = "H:\Anaconda3\envs\py310_torch\python.exe"

$arguments = @(
    "main.py",
    "--output_dir", "checkpoint\paper_v3_shared_base_seed123",
    "--log_dir", "log\paper_v3_shared_base_seed123.log",
    "--seed", "123",
    "--train_epochs", "60",
    "--warmup_epochs", "5",
    "--warmup_start_factor", "0.1",
    "--fusion_adapt_epochs", "0",
    "--early_stopping_patience", "10",
    "--early_stopping_min_delta", "0.001",
    "--rollout_steps", "4",
    "--validation_burn_in_steps", "4",
    "--train_batch_size", "4",
    "--eval_batch_size", "4",
    "--lr", "0.0006",
    "--min_lr", "0.00001",
    "--lambda_nll", "0.0001",
    "--velocity_loss_weight", "1.0",
    "--no-decoder_use_SEAN",
    "--no-decoder_use_GASF"
)

& $sggtPython @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Shared-base training failed with exit code $LASTEXITCODE"
}
