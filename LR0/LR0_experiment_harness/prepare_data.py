"""Create the exact stratified split specified in LR0; no model evaluation."""
import json
from pathlib import Path
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from harness import ROOT, sha256, write_json

def main():
    X, y = load_digits(return_X_y=True)
    idx = np.arange(len(y))
    rest, test = train_test_split(idx, test_size=0.2, stratify=y, random_state=0)
    train, val = train_test_split(rest, test_size=0.25, stratify=y[rest], random_state=0)
    split = {k: v.tolist() for k, v in [('train',train),('val',val),('test',test)]}
    assert [len(split[k]) for k in ['train','val','test']] == [1077,360,360]
    assert len(set(train) | set(val) | set(test)) == len(y)
    assert not (set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
    assert all(len(np.unique(y[v])) == 10 for v in split.values())
    dest = ROOT/'data'/'split.json'
    write_json(dest, split)
    write_json(ROOT/'data'/'manifest.json', {
        'source':'sklearn.datasets.load_digits', 'sklearn_version':sklearn.__version__,
        'records_total':len(y), 'excluded':0, 'features':64, 'classes':10,
        'split_seed':0, 'rule':'stratified test_size=0.2; val_size=0.25 of remainder; random_state=0 for both',
        'normalization':'X / 16 (known scale; no fitted statistics)',
        'files':{'data/split.json':{'sha256':sha256(dest),'sizes':{k:len(v) for k,v in split.items()}}},
        'raw_X_sha256':__import__('hashlib').sha256(X.tobytes()).hexdigest(),
        'raw_y_sha256':__import__('hashlib').sha256(y.tobytes()).hexdigest(),
    })
    print(json.dumps({k:len(v) for k,v in split.items()}))

if __name__ == '__main__': main()
