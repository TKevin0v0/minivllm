from pathlib import Path
import paramiko
key=paramiko.Ed25519Key.from_private_key_file(r'C:/Users/Lenovo/.ssh/id_rsa_slai')
t=paramiko.Transport(('10.1.30.50',31404)); t.connect(); t.auth_publickey('naie',key); c=paramiko.SSHClient(); c._transport=t
cmd="hostname; npu-smi info 2>&1 | head -40; which python; python --version; find ~ -maxdepth 5 -type f -name config.json 2>/dev/null | head -40; find ~ -maxdepth 3 -type d -name 'vllm-v2-npu' 2>/dev/null"
_i,o,e=c.exec_command(cmd,timeout=180); text=o.read().decode(errors='replace')+e.read().decode(errors='replace'); Path('benchmark-results/npu/env-current.log').write_text('$ '+cmd+'\n'+text); print(text)
