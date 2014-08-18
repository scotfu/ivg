#coding=utf8
import os
import json

from werkzeug import secure_filename

from flask.views import View, MethodView
from flask import Flask, request, session, g, redirect, url_for, abort,render_template, flash


from . import app
from .utils import allowed_file, csv_handler, get_colletions, case_query

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
        return get_colletions()



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
        return case_query(case_name)
#        print case_query(case_name)
#        return '1'
    def post(self):
        return redirect(url_for('index'))            
       

        
        
        

        
            