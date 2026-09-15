from pathlib import Path
import paramiko
pw=Path('.codex/gpu_password.txt').read_text().strip()
t=paramiko.Transport(('connect.nmb2.seetacloud.com',40423)); t.connect(); t.auth_password('root',pw)
s=paramiko.SFTPClient.from_transport(t)
script = "import sys,time\nsys.path.insert(0,'/root/bench/vllm-v2-npu')\nfrom src.engine.engine import Engine\nfrom src.config import EngineConfig\ne=Engine(EngineConfig(model='/root/models/models/Qwen--Qwen3-1.7B/snapshots/master',device='cuda'))\nt0=time.perf_counter(); first=None; n=0\nfor _ in e.generate('Hello, my name is',max_new_tokens=32,temperature=0.0):\n n+=1\n if first is None: first=time.perf_counter()\nt1=time.perf_counter()\nprint(f'tokens={n} ttft_ms={(first-t0)*1000:.3f} ttot_ms={(t1-t0)*1000:.3f} decode_tps={n/(t1-first):.3f}')\n"
f=s.open('/root/bench/latency.py','w'); f.write(script); f.close(); s.close()
c=paramiko.SSHClient(); c._transport=t; cmd='/root/miniconda3/bin/python3.12 /root/bench/latency.py'; _i,o,e=c.exec_command(cmd,timeout=900)
text=o.read().decode(errors='replace')+e.read().decode(errors='replace'); Path('benchmark-results/gpu-new/latency.log').write_text('$ '+cmd+'\n'+text); print(text)
