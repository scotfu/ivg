# !/usr/bin/env python

data = open('sorted.2000')
out = open('coor.csv.2000','w')
for line in data:
    out.write(str(','.join([c for c in line if c!='\n'])+'\n'))
    
data.close()
out.close()
