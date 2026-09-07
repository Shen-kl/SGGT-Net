"""Launch the 200/50 association-aware CGIC development run on Windows."""

import os
from pathlib import Path
import subprocess


def quote(value):
    return str(value).replace("'", "''")


def main():
    root = Path(__file__).resolve().parents[1]
    log_dir = root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    pid_path = log_dir / "dev_assoc_cgic_200_50_runner.pid"
    runner = root / "scripts" / "run_association_cgic_dev.ps1"

    environment = dict(os.environ)
    path_value = environment.get("PATH", environment.get("Path", ""))
    environment = {
        key: value for key, value in environment.items()
        if key.lower() != "path"
    }
    environment["PATH"] = path_value
    argument_line = (
        "-NoProfile -ExecutionPolicy Bypass -File "
        f'"{runner}"'
    )
    bootstrap = (
        "$ErrorActionPreference='Stop'; "
        "$p=Start-Process -FilePath 'powershell.exe' "
        f"-ArgumentList '{argument_line}' "
        f"-WorkingDirectory '{quote(root)}' -WindowStyle Hidden "
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
    print({"pid": int(pid_path.read_text(encoding="ascii").strip())})


if __name__ == "__main__":
    main()
