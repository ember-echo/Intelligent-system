import argparse
from harness import load_config, run_config

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--seed',type=int)
    p.add_argument('--series',default='manual');p.add_argument('--eval-test',action='store_true')
    p.add_argument('--slot',help='Unique planned run key for resumable series')
    p.add_argument('--debug',action='store_true',help='Debug only; excluded from report')
    a=p.parse_args();config=load_config(a.config,a.seed)
    if a.slot:config['slot']=a.slot
    run_config(config,a.series,a.eval_test,debug=a.debug)

if __name__=='__main__': main()
