from flask import Flask,request
from flask_restful import Resource,Api
from MSPs_model import MultilayerPs
import torch

the_model = MultilayerPs()
the_model.load_state_dict(torch.load('Gender_Detection_MLPs.pth'))

values = {}
app = Flask(__name__)
api = Api(app)

class predict_gender(Resource):
    def get(self, pid):
        vals = [float(d) for d in values[pid].split(',') ]
        vals = torch.FloatTensor(vals)
        the_model.eval()
        y = the_model(vals)
        prediction = y.detach().numpy()
        prediction = prediction.round()[0]
        if (prediction == 0):
            return 'Male'
        if (prediction == 1):
            return 'Female'
        
    def put(self,pid):
        values[pid] = request.form['data']
        return {pid:values[pid]}

api.add_resource(predict_gender, '/<string:pid>')

if __name__ == '__main__':
    app.run(debug=True)
