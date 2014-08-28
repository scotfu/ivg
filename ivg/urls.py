from . import app
from .views import CaseView, CaseQueryView, UploadView, IndexView, KmeansView, AggregationView


app.add_url_rule('/', view_func=IndexView.as_view('index'))
app.add_url_rule('/upload/',view_func=UploadView.as_view('upload'))
app.add_url_rule('/case/<case_name>/', view_func=CaseView.as_view('case'))
app.add_url_rule('/case/<case_name>/query/', view_func=CaseQueryView.as_view('case_query'))
app.add_url_rule('/case/<case_name>/aggregation/', view_func=AggregationView.as_view('aggregation'))
app.add_url_rule('/case/<case_name>/kmeans/<step>/', view_func=KmeansView.as_view('kmean_query'))