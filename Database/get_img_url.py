#coding:utf8

"""Print urls of images related to a certain
theme according to Duckduckgo search engine.

This script might be useful when chained with
a command line program like wget in linux:

 python get_img_url.py luxury cars | wget -P Database/
"""

import requests
import sys

if len(sys.argv) == 1:
    raise ValueError("Pass theme as argument")

query = "+".join(sys.argv[1:])
r = requests.get("https://duckduckgo.com/i.js?q=" + query)

for res in r.json()['results']:
    print res['image']
