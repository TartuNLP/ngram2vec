#!/usr/bin/env python3

import re

def file2snts(filename):
	with open(filename, 'r') as fh:
		for line in fh:
			raw = [rawTok.split("|") for rawTok in line.strip().lower().split(" ")]
			yield [w for w, t in raw if re.search(r'\w', w)]
