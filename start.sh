#! /bin/bash
uwsgi -s uwsgi.sock --module wsgi --callable app --master --processes 4 --threads 2 --stats 127.0.0.1:5001

