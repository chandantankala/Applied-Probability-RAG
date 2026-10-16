"""Start both local services, run integration checks, and stop them afterward."""
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    project = Path(__file__).resolve().parents[1]
    processes, handles = [], []
    for port in (8000, 8001):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise SystemExit(f"Port {port} is in use. Stop your existing MathPaper services before running this check.")
    with tempfile.TemporaryDirectory() as directory:
        try:
            for name, command, port in [("api", "serve", 8000), ("mcp", "mcp", 8001)]:
                log = (Path(directory)/(name+".log")).open("w")
                handles.append(log)
                process = subprocess.Popen([sys.executable, "-m", "mathpaper", command], cwd=project, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
                processes.append(process)
                deadline = time.monotonic()+20
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(name+" service failed to start")
                    with socket.socket() as sock:
                        if sock.connect_ex(("127.0.0.1", port)) == 0:
                            break
                    time.sleep(.1)
                else:
                    raise RuntimeError(name+" service startup timed out")
            subprocess.run([sys.executable, "-u", str(project/"evaluation/smoke.py")], cwd=project, check=True)
        except Exception:
            for handle in handles:
                handle.flush()
                print(Path(handle.name).read_text()[-4000:], file=sys.stderr)
            raise
        finally:
            for process in processes:
                process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for handle in handles:
                handle.close()


if __name__ == "__main__":
    main()
