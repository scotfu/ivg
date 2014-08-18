from . import app
from .views import CaseView, CaseQueryView, UploadView, IndexView


app.add_url_rule('/', view_func=IndexView.as_view('index'))
app.add_url_rule('/upload/',view_func=UploadView.as_view('upload'))
app.add_url_rule('/case/<case_name>/', view_func=CaseView.as_view('case'))
app.add_url_rule('/case/<case_name>/query/', view_func=CaseQueryView.as_view('case_query'))