#!/usr/bin/python

import os
import glob
import time
import pathlib
import shutil

max_len = 20
for i in range(60 * 48):
    bad_files = glob.glob(".darts/checkpoints/*") + glob.glob(".darts/untrained_models/*")
    bad_files = sorted(bad_files)
    print(bad_files)
    if len(bad_files) > max_len:
        for fn in bad_files[max_len:]:
            #os.remove(fn)
            #file_to_rem = pathlib.Path(fn)
            #file_to_rem.unlink()
            try:
                shutil.rmtree(fn)
                print("remove: ", fn, flush=True)
            except:
                pass
    print("Waiting one minute", flush=True)
    time.sleep(60) 