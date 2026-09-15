from pathlib import Path
import paramiko

root = Path('.').resolve()
pw = Path('.codex/gpu_password.txt').read_text(encoding='utf-8').strip()
t = paramiko.Transport(('connect.nmb2.seetacloud.com', 32942)); t.connect(); t.auth_password('root', pw)
s = paramiko.SFTPClient.from_transport(t)
def put_tree(local, remote):
    for p in local.rglob('*'):
        if p.is_dir() or '__pycache__' in p.parts or p.suffix in {'.pyc'}:
            continue
        rel = p.relative_to(local).as_posix(); target = remote + '/' + rel
        parent = '/'.join(target.split('/')[:-1])
        cur = ''
        for part in parent.split('/'):
            if not part: continue
            cur += '/' + part
            try: s.stat(cur)
            except IOError: s.mkdir(cur)
        s.put(str(p), target)
put_tree(root / 'vllm-v2', '/root/bench/vllm-v2-modified')
put_tree(root / 'qwen3-1.7b-ms', '/root/models/Qwen3-1.7B')
s.close(); t.close()
print('uploaded vllm-v2-modified and Qwen3-1.7B')
