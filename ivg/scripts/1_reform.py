# !/usr/bin/env python
#coding=utf8
import os
import csv
BUFFER_SIZE = 100

def transform(file_name):
    c_file = open(file_name)
    out_file = open('raw_data','wr')
    reader = csv.reader(c_file)
    h = reader.next()
    writer = csv.writer(out_file, delimiter=',')
    writer.writerow(['id','country','age','gender','education','bitstring','suggested_priority'])
    lineno = 0
    for row in reader:
        try:
            bit_string = transform_pri_to_bits([row[i] for i in range(7,13)])
            data= [lineno] + [row[i] for i in range(3,7)] + [bit_string] + [row[13]]
            lineno += 1
            writer.writerow(data)
        except ValueError:
            print lineno
        except Exception,e:
            print row
            print lineno,e
    c_file.close()
    out_file.close()
    
def transform_pri_to_bits(six):
    s = ['0'] * 16 # a list of 0s
    for p in six:
        try:
            p = int(p) - 100
            s[p] = '1'
        except ValueError:
            print 'Found one record with less than six priorities'
            raise
        except Exception,e:
            print e
            
    return ''.join(s)    
    

if __name__ == '__main__':
    import sys
    file_name = sys.argv[1]
    transform(file_name)
