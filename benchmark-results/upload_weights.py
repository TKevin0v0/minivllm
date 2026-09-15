from pathlib import Path
import paramiko
root=Path('.').resolve(); pw=(root/'.codex/gpu_password.txt').read_text(encoding='utf-8').strip()
t=paramiko.Transport(('connect.nmb2.seetacloud.com',32942)); t.connect(); t.auth_password('root',pw); s=paramiko.SFTPClient.from_transport(t)
local=root/'qwen3-1.7b-ms'; remote='/root/models/Qwen3-1.7B'
for p in local.rglob('*'):
    if p.is_dir(): continue
    target=remote+'/'+p.relative_to(local).as_posix(); parent='/'.join(target.split('/')[:-1]); cur=''
    for part in parent.split('/'):
        if not part: continue
        cur+='/'+part
        try: s.stat(cur)
        except IOError: s.mkdir(cur)
    s.put(str(p),target)
s.close(); t.close(); print('uploaded Qwen3-1.7B')
