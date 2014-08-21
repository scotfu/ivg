#coding=utf8
import os
import json

from werkzeug import secure_filename

from flask.views import View, MethodView
from flask import Flask, request, session, g, redirect, url_for, abort,render_template, flash


from . import app
from .utils import allowed_file, csv_handler, get_collections, case_query, kmeans_query, kmeans2_query, get_all_points, CustomEncoder

class ListView(View):

    def get_template_name(self):

        raise NotImplementedError()
        
    def render_template(self, context):

        return render_template(self.get_template_name(), **context)

    def dispatch_request(self,**kwargs):
        context = {'objects': self.get_objects()}
        return self.render_template(context)

        
class CaseView(ListView):

    def get_template_name(self):
        return 'chart.html'

    def get_objects(self):
        return None

        
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
            collection_name = request.form.get('url')
            file.save(file_path)
            csv_handler(file_path, collection_name)
            return redirect(url_for('index'))
        flash('Upload failed, please try again')
        return redirect(url_for('upload'))            
       

class CaseQueryView(MethodView):

    def get(self,case_name):
        algorithm = request.args.get('algorithm', 'nmds')
        data = {'data':{'groups':[], 'margin':[]}
                ,'success':'false'}
                
        ec = CustomEncoder()
        data['data']['groups'].append(case_query(case_name, algorithm))
        data['success'] = 'true'
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
        
        all_points = get_all_points(case_name, algorithm)

        if step == '1':
            
            cluster_matrix, centroids = kmeans_query(case_name, selected_points, algorithm)
            group_data = [[] for i in range(len(centroids))]
            print cluster_matrix
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
            return json.dumps(data)
        else:
            
            #get result of first step k-means
            cluster_matrix, centroids, all_points = kmeans2_query(case_name,selected_points, second_selected_points, algorithm)
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
            return json.dumps(data)
            

    def post(self):
        return redirect(url_for('index'))            

        

        
        
        

        
            