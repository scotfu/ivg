#coding=utf8
import os
import json

from werkzeug import secure_filename

from flask.views import View, MethodView
from flask import Flask, request, session, g, redirect, url_for, abort,render_template, flash


from . import app
from .utils import allowed_file, get_collections, case_query, kmeans_query, kmeans2_query, get_all_points, aggregation, histogram, CustomEncoder, get_case_info

from .tasks import csv_handler

class ListView(View):

    def get_template_name(self):

        raise NotImplementedError()
        
    def render_template(self, context):

        return render_template(self.get_template_name(), **context)

    def dispatch_request(self,**kwargs):
        context = {'objects': self.get_objects()}
        return self.render_template(context)

        
class CaseView(View):

    def get_template_name(self):
        return 'chart.html'

    def render_template(self, context):

        return render_template(self.get_template_name(), **context)

    def dispatch_request(self,**kwargs):
        url, content = get_case_info(kwargs.get('case_name','hi'))
        context = {'url':url, 'content': content}
        return self.render_template(context)


        
class IndexView(ListView):

    def get_template_name(self):
        return 'index.html'

    def get_objects(self):
        return get_collections()



class UploadView(MethodView):

    def get(self):
        return render_template('upload.html')            
            
    def post(self):
        file = request.files['file']
        if file and allowed_file(file.filename):
            file_name = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            name = request.form.get('name').strip()
            url = request.form.get('url').strip()
            content = request.form.get('content')
            file.save(file_path)
            csv_handler.delay(file_path, name, url, content)
            return redirect(url_for('index'))
        flash('Upload failed, please try again')
        return redirect(url_for('upload'))            
       

class CaseQueryView(MethodView):

    def get(self,case_name):
        algorithm = request.args.get('algorithm', 'nmds')
        data = {'data':{'groups':[], 'margin':[]}
                ,'success':'false'}
                
        ec = CustomEncoder()
        all_points, margin = case_query(case_name, algorithm)
        data['data']['groups'].append(all_points)
        data['success'] = 'true'
        data['data']['margin'] =margin
        return ec.encode(data)
    def post(self):
        return redirect(url_for('index'))            

        
class AggregationView(MethodView):

    def get(self,case_name):
        ec = CustomEncoder()
        points = list(set(request.values.getlist("id")))
        points, header, mins, maxs = aggregation(case_name, points)
        points = zip(header,points)
        data = {'dimensions':points, 'max':maxs, 'min':mins}
        return ec.encode(data)
    def post(self):
        return redirect(url_for('index'))            


class HistogramView(MethodView):

    def get(self,case_name,dimension):
        ec = CustomEncoder()
        counts,bins = histogram(case_name,int(dimension))
        data = {'counts':counts, 'bins':bins}
        return ec.encode(data)
    def post(self):
        return redirect(url_for('index'))            
        
        
class KmeansView(MethodView):

    def get(self,case_name,step):
        selected_points = list(set(request.values.getlist("id")))
        second_selected_points = list(set(request.values.getlist("second_id")))
        algorithm = request.args.get('algorithm','nmds')
        data = {'data':{'groups':[], 'margin':[]}
                ,'success':'false'}
        
#        all_points = get_all_points(case_name, algorithm)

        if step == '1':
            
            cluster_matrix, centroids, all_points, margin = kmeans_query(case_name, selected_points, algorithm)
            group_data = [[] for i in range(len(centroids))]
            for point_pos in range(len(all_points)):
                k = 0
                while True:
                    if cluster_matrix[k][point_pos] == 1:
                        break
                    else:
                        k += 1
                group_data[k].append(all_points[point_pos])
            data['data']['groups'] = group_data
            data['data']['margin'] = margin
            data['success'] = 'true'
            return json.dumps(data)
        else:
            
            #get result of first step k-means
            cluster_matrix, centroids, all_points, margin = kmeans2_query(case_name,selected_points, second_selected_points, algorithm)
            group_data = [[] for i in range(len(centroids))]

            for point_pos in range(len(all_points)):
                k = 0
                while True:
                    if cluster_matrix[k][point_pos] == 1:
                        break
                    else:
                        k += 1
                group_data[k].append(all_points[point_pos])
            data['data']['groups'] = group_data
            data['success'] = 'true'
            data['data']['margin'] = margin
            return json.dumps(data)
            

    def post(self):
        return redirect(url_for('index'))
