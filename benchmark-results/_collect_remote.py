from pathlib import Path
import paramiko

ROOT = Path(__file__).parent
password = Path(__file__).parents[1] / ".codex" / "modelmate_password.txt"
secret = password.read_text(encoding="utf-8").strip()

targets = {
    "gpu": ("connect.nmb2.seetacloud.com", 32942, "root"),
    "npu": ("10.1.30.50", 31404, "naie"),
}
commands = {
    "env.log": "nvidia-smi 2>&1; python --version 2>&1; python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())' 2>&1",
    "dirs.log": "pwd; find ~ -maxdepth 3 -type d \( -iname '*qwen*' -o -iname '*vllm*' \) 2>/dev/null | head -80",
}

for name, (host, port, user) in targets.items():
    outdir = ROOT / name
    outdir.mkdir(exist_ok=True)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host, port=port, username=user, password=secret, timeout=20, banner_timeout=20, auth_timeout=20)
        for filename, command in commands.items():
            stdin, stdout, stderr = client.exec_command(command, timeout=60)
            text = stdout.read().decode(errors="replace")
            err = stderr.read().decode(errors="replace")
            (outdir / filename).write_text(f"$ {command}\n{text}{err}", encoding="utf-8")
    except Exception as exc:
        (outdir / "connection-failure.log").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
    finally:
        client.close()
