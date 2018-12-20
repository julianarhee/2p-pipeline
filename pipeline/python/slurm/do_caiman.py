#!/usr/local/bin/python

import argparse
import uuid
import sys
import os
import commands
import json

parser = argparse.ArgumentParser(
    description = '''Look for XID files in session directory.\nFor PID files, run tiff-processing and evaluate.\nFor RID files, wait for PIDs to finish (if applicable) and then extract ROIs and evaluate.\n''',
    epilog = '''AUTHOR:\n\tJuliana Rhee''')
parser.add_argument('-i', '--animalid', dest='animalid', action='store', default='', help='Animal ID')
parser.add_argument('-S', '--session', dest='session', action='store',  default='', help='session (fmt: YYYYMMDD)')
parser.add_argument('-A', '--acquisition', dest='acquisition', action='store',  default='', help='acquisition folder')
parser.add_argument('-R', '--run', dest='run', action='store',  default='', help='run folder')
parser.add_argument('-p', '--pid', dest='pidhash', action='store',  default='', help='6-char PID hash')

args = parser.parse_args()


