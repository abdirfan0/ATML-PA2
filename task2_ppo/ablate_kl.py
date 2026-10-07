from __future__ import annotations
import argparse
import subprocess
import sys
from common.data import load_yaml


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/ppo.yaml')
    p.add_argument('--stage',choices=['train','evaluate','all'],default='all')
    a=p.parse_args();cfg=load_yaml(a.config)
    for beta in cfg['kl_values']:
        name=f'kl_{float(beta):.2f}'.replace('.','p')
        out=f'outputs/task2_ppo/{name}'
        if a.stage in ['train','all']:
            subprocess.run([sys.executable,'-u','-m','task2_ppo.continue_train',
                '--config',a.config,'--run-name',name,'--output',out,
                '--updates',str(cfg['fork_updates']),'--kl-beta',str(beta)],check=True)
        if a.stage in ['evaluate','all']:
            subprocess.run([sys.executable,'-u','-m','task2_ppo.evaluate',
                '--config',a.config,'--adapter',out,'--name',name],check=True)

if __name__=='__main__':main()
