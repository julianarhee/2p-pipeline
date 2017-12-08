#!/usr/bin/env python2
import os

#os.system('python batch_process_ids.py \
#            -iJR063 \
#            -S20171128_JR063_test4 \
#            --flyback \
#            -F1 \
#            --bidi \
#            --motion')

# os.system('python process_pids_for_session.py -iJR063 -S20171128_JR063_test5')

os.system('python ../process_pids_for_session.py -iJR063 -S20171128_JR063_testbig -n12')           

