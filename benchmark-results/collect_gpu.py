from pathlib import Path
import paramiko

pw = Path('.codex/gpu_password.txt').read_text(encoding='utf-8').strip()
t = paramiko.Transport(('connect.nmb2.seetacloud.com', 32942))
t.connect()
t.auth_password('root', pw)
c = paramiko.SSHClient()
c._transport = t
commands = {
    'env.log': 'nvidia-smi; /root/miniconda3/bin/python --version',
    'torch.log': "/root/miniconda3/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'",
    'models.log': "find /root/autodl-fs /root/autodl-pub /root/autodl-tmp -maxdepth 6 -type f -name config.json 2>/dev/null | head -100",
}
outdir = Path('benchmark-results/gpu')
for filename, command in commands.items():
    _i, out, err = c.exec_command(command, timeout=180)
    text = out.read().decode(errors='replace') + err.read().decode(errors='replace')
    (outdir / filename).write_text('$ ' + command + '\n' + text, encoding='utf-8')
    print(text)
c.close()
