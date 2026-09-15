from pathlib import Path
import paramiko
pw=Path('.codex/gpu_password.txt').read_text().strip(); t=paramiko.Transport(('connect.nmb2.seetacloud.com',40423)); t.connect(); t.auth_password('root',pw); c=paramiko.SSHClient(); c._transport=t
base='/root/bench/vllm-v2-npu'; model='/root/models/models/Qwen--Qwen3-1.7B/snapshots/master'; py='/root/miniconda3/bin/python3.12'
cmd=f"cd {base} && PYTHONPATH=/root/miniconda3/lib/python3.12/site-packages:{base} {py} run.py --model {model} --device cuda --max-new-tokens 32 --prompt 'Hello, my name is'"
_i,o,e=c.exec_command(cmd,timeout=900); text=o.read().decode(errors='replace')+e.read().decode(errors='replace'); Path('benchmark-results/gpu-new/test-single-2.log').write_text('$ '+cmd+'\n'+text); print(text)
