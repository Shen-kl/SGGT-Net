"""Launch the formal innovation runner through a clean PowerShell bootstrap.

The hosted shell can expose both ``Path`` and ``PATH``.  ``Start-Process``
rejects that duplicate-key environment, while a direct child process can be
reaped with the tool job.  Python first normalizes the environment, then a
short-lived PowerShell bootstrap creates the independent hidden runner.
"""

import os
from pathlib import Path
import subprocess


def ps_quote(value: Path) -> str:
    """Quote a literal path for a single-quoted PowerShell string."""

    return str(value).replace("'", "''")


def main():
    root = Path(__file__).resolve().parents[1]
    stdout_path = root / "log" / "formal_innovation_v1_runner.stdout.log"
    stderr_path = root / "log" / "formal_innovation_v1_runner.stderr.log"
    pid_path = root / "log" / "formal_innovation_v1_runner.pid"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    path_value = environment.get("PATH", environment.get("Path", ""))
    environment = {
        key: value for key, value in environment.items()
        if key.lower() != "path"
    }
    environment["PATH"] = path_value
    runner_path = root / "scripts" / "run_formal_innovation_multiseed.ps1"
    argument_line = (
        "-NoProfile -ExecutionPolicy Bypass -File "
        f"\"{runner_path}\""
    )
    bootstrap = (
        "$ErrorActionPreference = 'Stop'; "
        "$process = Start-Process "
        "-FilePath 'powershell.exe' "
        f"-ArgumentList '{argument_line.replace(chr(39), chr(39) * 2)}' "
        f"-WorkingDirectory '{ps_quote(root)}' "
        "-WindowStyle Hidden -PassThru; "
        f"Set-Content -LiteralPath '{ps_quote(pid_path)}' "
        "-Value $process.Id -Encoding ascii; "
        "Write-Output $process.Id"
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
    if completed.returncode != 0:
        raise SystemExit(completed.stderr.strip() or completed.stdout.strip())
    pid = int(pid_path.read_text(encoding="ascii").strip())
    print({"pid": pid, "bootstrap_returncode": completed.returncode})


if __name__ == "__main__":
    main()
