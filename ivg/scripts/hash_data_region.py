# /usr/bin/env python
#coding=utf-8
import csv


def hash_data():
    hash_dict={}
    data = open('data')
    reader = csv.reader(data)
    header = reader.next()
    for row in reader:
        try:
            hash_dict[row[5]]['count'] += 1
            hash_dict[row[5]]['ids'].add(row[1])
        except Exception,e:
            hash_dict[row[5]] = {'count':1,'ids':set(row[1])}

                      
    import pprint         
    out = open('data_count_by_region','w')
    writer = csv.writer(out)
    for key in hash_dict.keys():
        writer.writerow([key, hash_dict[key]['count'], hash_dict[key]['ids']])
    data.close()
    out.close()
if __name__ == '__main__':
    hash_data()
