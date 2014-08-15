# /usr/bin/env python
#coding=utf-8
import csv


def hash_data():
    hash_dict={}
    data = open('raw_data')
    reader = csv.reader(data)
    header = reader.next()
    for row in reader:
        try:
            hash_dict[row[5]]['count'] += 1
            hash_dict[row[5]]['ids'].append(row[0])
        except:
            hash_dict[row[5]] = {'count':1,'ids':[row[0]]}
                      
    import pprint         
    out = open('data_count','w')
    writer = csv.writer(out)
    for key in hash_dict.keys():
        writer.writerow([key,hash_dict[key]['count'],hash_dict[key]['ids']])
    data.close()
    out.close()
if __name__ == '__main__':
    hash_data()
