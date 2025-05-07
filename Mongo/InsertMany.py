"""
generates JSON to insert
"""
from os import listdir
from os.path import isfile, join


path = "/Users/aidanlear/Desktop/TestImages"
files = [f for f in listdir(path) if isfile(join(path, f))]

print(files[:10])



