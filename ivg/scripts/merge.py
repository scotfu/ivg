#!/usr/bin/evn python
f1 = open('r_data_count.1000').readlines()
f2 = open('tmp_data').readlines()
f3 = open('x','w')
l = 0
while l<1000:
	for country in f1[l].strip().split(','):
		f3.write(f2[l].strip()+' '+country.strip().replace("'",'' )+'\n')
	l+=1	