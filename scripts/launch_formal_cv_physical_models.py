"""Launch the formal CV physical-model runner without a visible window."""

import os
from pathlib import Path
import subprocess
import time


def main():
    root = Path(__file__).resolve().parents[1]
    stdout_path = root / "log" / "formal_cv_physical_runner.stdout.log"
    stderr_path = root / "log" / "formal_cv_physical_runner.stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    # The hosting shell can expose both Path and PATH. Python normalizes this
    # mapping before CreateProcess, avoiding Start-Process's duplicate-key bug.
    path_value = environment.get("PATH", environment.get("Path", ""))
    environment = {
        key: value
        for key, value in environment.items()
        if key.lower() != "path"
    }
    environment["PATH"] = path_value
    with stdout_path.open("ab", buffering=0) as stdout, stderr_path.open(
        "ab", buffering=0
    ) as stderr:
        process = subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(root / "scripts" / "run_formal_cv_physical_models.ps1"),
            ],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            env=environment,
            creationflags=(
                subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NEW_PROCESS_GROUP
            ),
            close_fds=True,
        )
    time.sleep(2.0)
    print(process.pid, process.poll())


if __name__ == "__main__":
    main()
