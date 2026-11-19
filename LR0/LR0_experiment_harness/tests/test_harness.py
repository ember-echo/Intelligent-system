import copy
import subprocess
import tempfile
import unittest
from pathlib import Path
from harness import ROOT, accuracy, git_commit, load_config, read_data, validate_config

class HarnessTests(unittest.TestCase):
    def test_known_accuracy(self):
        self.assertEqual(accuracy([1,2,3,4],[1,2,3,3]),0.75)
        with self.assertRaises(ValueError): accuracy([],[])
    def test_exact_split_and_hashes(self):
        X,y,split=read_data();self.assertEqual([len(split[k]) for k in ['train','val','test']],[1077,360,360])
        self.assertEqual(tuple(X.shape),(1797,64));self.assertGreaterEqual(float(X.min()),0);self.assertLessEqual(float(X.max()),1)
    def test_negative_learning_rate_rejected(self):
        c=load_config(ROOT/'configs/base.yaml');c['train']['lr']=-1
        with self.assertRaisesRegex(AssertionError,'lr must be positive'):validate_config(c)
    def test_dirty_git_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            def git(*args): return subprocess.run(['git',*args],cwd=root,check=True,capture_output=True)
            git('init');(root/'code.py').write_text('x=1\n');git('add','code.py')
            git('-c','user.name=LR0 Test','-c','user.email=LR0_experiment_harness-test@example.invalid','commit','-m','test')
            self.assertEqual(len(git_commit(root)),40)
            (root/'code.py').write_text('x=2\n')
            with self.assertRaisesRegex(RuntimeError,'Uncommitted'):git_commit(root)
            git('checkout','--','code.py');(root/'new.py').write_text('x=3\n')
            with self.assertRaisesRegex(RuntimeError,'Uncommitted'):git_commit(root)

if __name__=='__main__':unittest.main()
