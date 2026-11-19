"""Relocate only artifact URIs after moving this self-contained submission."""
import shutil
import sqlite3
from datetime import datetime
from harness import ROOT

def main():
    db=ROOT/'mlflow.db'
    backup=db.with_name('mlflow.backup-'+datetime.now().strftime('%Y%m%d%H%M%S')+'.db')
    shutil.copy2(db,backup)
    root_uri=(ROOT/'mlartifacts').as_uri()
    with sqlite3.connect(db) as c:
        c.execute('UPDATE experiments SET artifact_location=? WHERE name=?',(root_uri,'LR0_experiment_harness'))
        exp=c.execute('SELECT experiment_id FROM experiments WHERE name=?',('LR0_experiment_harness',)).fetchone()[0]
        c.execute('UPDATE runs SET artifact_uri=? || run_uuid || ? WHERE experiment_id=?',(root_uri+'/', '/artifacts',exp))
    print('Updated artifact URIs. Original database backup:',backup.name)

if __name__=='__main__':main()
