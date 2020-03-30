from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from model_wrapper import ModelWrapper, CustomTransformer
import pandas as pd


classifier = ModelWrapper()
classifier.load_model('svc_20032020')

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('query')


class ReviewResponder(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']

        # pack review in DataFrame

        review = pd.DataFrame({'Review': [user_query]})
        rev_pred = classifier.predict(review)[0]

        return {'response': 'Thanks for review!' if rev_pred == 1 else 'Sorry, we\'ll try to do better'}


# review = pd.DataFrame({'Review': ['Big portions and very tasty']})

# rev_pred = classifier.predict(review)[0]
# print('Thanks for review!' if rev_pred == 1 else 'Sorry, we\'ll try to do better')

api.add_resource(ReviewResponder, '/')

if __name__ == '__main__':
    app.run(debug=True)
