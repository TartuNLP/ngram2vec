#!/usr/bin/env python3

import sys

with open(sys.argv[1], 'r') as fh:
	for line in fh:
		print(" ".join([wt.split("|")[0] for wt in line.strip().lower().split(" ")]))
