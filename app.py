from flask import Flask, request
from flask_cors import CORS
from flask_restplus import Api, Resource, reqparse
from services.classifier import find, add_path
from services.convert_file_to_text import read_file


def default_doc():
    return {
        'responses': {
            '200': 'Success',
            '500': 'Internal Failure',
            '503': 'Service Unavailable'
        }}


app = Flask(__name__, static_url_path='/app', static_folder='web/drop')
CORS(app)

api = Api(
    app,
    version='1.0',
    title='Our awesome api',
)

api_ns = api.namespace('api', description='Api operations')

classification_parser = reqparse.RequestParser()
classification_parser.add_argument('text', type=str, default=None)


@api_ns.route('/main')
class OurResource(Resource):
    @classmethod
    @api.doc(**default_doc())
    @api.expect(classification_parser)
    def get(cls):
        arguments = classification_parser.parse_args()
        if arguments['text'] is None:
            return {'error': 'missing text'}, 503
        content = arguments['text']

        paths, i = find(content)
        return {'input': content, 'paths': paths, 'id': i}, 200


@api_ns.route('/file')
class FileResource(Resource):
    @classmethod
    @api.doc(**default_doc())
    def post(cls):
        file = request.files.get('file')
        if file is None:
            return {'error': 'missing file'}, 503

        content = read_file(file)

        paths, i = find(content)
        return {'input': content, 'paths': paths, 'id': i}, 200


add_path_parser = reqparse.RequestParser()
add_path_parser.add_argument('path', type=str, default=None)
add_path_parser.add_argument('id', type=int, default=None)


@api_ns.route('/add_path')
class AddPath(Resource):
    @classmethod
    @api.doc(**default_doc())
    @api.expect(add_path_parser)
    def post(cls):
        arguments = add_path_parser.parse_args()
        id_ = arguments.get('id')
        path = arguments.get('path')
        if id_ is None:
            return {'error': 'missing id'}, 503
        if path is None:
            return {'error': 'missing path'}, 503
        add_path(id_, path)
        return {'result': True}, 200
