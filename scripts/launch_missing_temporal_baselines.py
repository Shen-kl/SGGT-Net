"""Launch the missing seed-789 Temporal baseline as a hidden Windows job."""

import os
from pathlib import Path
import subprocess


def quote(value):
    return str(value).replace("'", "''")


def main():
    root = Path(__file__).resolve().parents[1]
    log_dir = root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    pid_path = log_dir / "missing_temporal_seed789_runner.pid"
    stdout_path = log_dir / "missing_temporal_seed789_runner.stdout.log"
    stderr_path = log_dir / "missing_temporal_seed789_runner.stderr.log"
    runner = root / "scripts" / "run_missing_temporal_baselines.ps1"

    environment = dict(os.environ)
    path_value = environment.get("PATH", environment.get("Path", ""))
    environment = {
        key: value for key, value in environment.items()
        if key.lower() != "path"
    }
    environment["PATH"] = path_value
    argument_line = (
        "-NoProfile -ExecutionPolicy Bypass -File "
        f'"{runner}" -Seeds 789'
    )
    bootstrap = (
        "$ErrorActionPreference='Stop'; "
        "$p=Start-Process -FilePath 'powershell.exe' "
        f"-ArgumentList '{quote(argument_line)}' "
        f"-WorkingDirectory '{quote(root)}' -WindowStyle Hidden "
        f"-RedirectStandardOutput '{quote(stdout_path)}' "
        f"-RedirectStandardError '{quote(stderr_path)}' "
        "-PassThru; "
        f"Set-Content -LiteralPath '{quote(pid_path)}' "
        "-Value $p.Id -Encoding ascii; $p.Id"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", bootstrap],
        cwd=root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=environment,
        creationflags=subprocess.CREATE_NO_WINDOW,
        check=False,
    )
    if completed.returncode:
        raise SystemExit(completed.stderr.strip() or completed.stdout.strip())
    print({
        "pid": int(pid_path.read_text(encoding="ascii").strip()),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    })


if __name__ == "__main__":
    main()
