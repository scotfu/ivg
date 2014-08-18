import os


from flask import Flask

app = Flask(__name__)

app.secret_key = 'A0sdfhasd~Zr98j/3yX R~XHH!jmN]LWX/,?RT'



app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path,'media')
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'csv'])


from . import views
from . import urls

if __name__ == '__main__':
    app.run()
