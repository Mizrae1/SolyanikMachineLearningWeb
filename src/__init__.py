from flask.json import jsonify
from flask import Flask, config, redirect, render_template, request, redirect, url_for
import os
from os.path import join, dirname, realpath
from src.constants.http_status_codes import HTTP_404_NOT_FOUND, HTTP_500_INTERNAL_SERVER_ERROR
from src.auth import auth
from src.bookmarks import bookmarks
from src.database import db, Bookmark
from src.config.swagger import template, swagger_config
import src.feature_extraction
from flask_jwt_extended import JWTManager
from flasgger import Swagger, swag_from
import mysql.connector
import pandas as pd
import csv
import numpy as np
import _pickle as cPickle

def create_app(test_config=None):

    app = Flask(__name__, instance_relative_config=True)
    global finaly_result
    finaly_result = []
    global unit_to_mult
    unit_to_mult = []
    if test_config is None:
        app.config.from_mapping(
            SECRET_KEY=os.environ.get("SECRET_KEY"),
            SQLALCHEMY_DATABASE_URI=os.environ.get("SQLALCHEMY_DB_URI"),
            SQLALCHEMY_TRACK_MODIFICATIONS=False,
            JWT_SECRET_KEY=os.environ.get('JWT_SECRET_KEY'),


            SWAGGER={
                'title': "Analysis API",
                'uiversion': 3
            }
        )
    else:
        app.config.from_mapping(test_config)

    db.app = app
    db.init_app(app)

    JWTManager(app)
    app.register_blueprint(auth)
    app.register_blueprint(bookmarks)

    app.config["DEBUG"] = True

    # Upload folder
    UPLOAD_FOLDER = 'src/static/files'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


    # Database
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="csvdata"
    )

    mycursor = mydb.cursor(buffered=True)

    mycursor.execute("SHOW DATABASES")

    Swagger(app, config=swagger_config, template=template)

    @app.get('/<short_url>')
    @swag_from('./docs/short_url.yaml')
    def redirect_to_url(short_url):
        bookmark = Bookmark.query.filter_by(short_url=short_url).first_or_404()

        if bookmark:
            bookmark.visits = bookmark.visits+1
            db.session.commit()
            return redirect(bookmark.url)

    @app.errorhandler(HTTP_404_NOT_FOUND)
    def handle_404(e):
        return jsonify({'error': 'Not found'}), HTTP_404_NOT_FOUND

    @app.errorhandler(HTTP_500_INTERNAL_SERVER_ERROR)
    def handle_500(e):
        return jsonify({'error': 'Что-то пошло не так, мы работаем над этим'}), HTTP_500_INTERNAL_SERVER_ERROR

    @app.route('/download')
    def index():
        # Set The upload HTML template '\templates\index.html'
        return render_template('index.html')

    @app.route('/version')
    def version():
        # Set The upload HTML template '\templates\index.html'
        return render_template('version.html')

    @app.route('/info')
    def info():
        # Set The upload HTML template '\templates\index.html'
        return render_template('info.html')

    @app.route('/autors')
    def autors():
        # Set The upload HTML template '\templates\index.html'
        return render_template('autors.html')

    @app.route('/privacy')
    def privacy():
        # Set The upload HTML template '\templates\index.html'
        return render_template('privacy.html')

    @app.route('/result')
    def res():
        global finaly_result
        global unit_to_mult
        unt = unit_to_mult
        res = finaly_result
        # Set The upload HTML template '\templates\index.html'
        return render_template('result.html', res=res,unt=unt)

    # Get the uploaded files
    @app.route("/download", methods=['POST'])
    @swag_from('./docs/upload.yaml')
    def uploadFiles():
        # get the uploaded file
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            # set the file path
            uploaded_file.save(file_path)
            parseCSV(file_path, uploaded_file.filename)
        # save the file
        return redirect(url_for('index'))

    def parseCSV(filePath, file_name):
        # CVS Column Names
        col_names = ['test_X', 'test_Y', 'train_X', 'train_Y']
        # Use Pandas to parse the CSV file
        csvData = pd.read_csv(filePath, names=col_names, header=None)
        # Loop through the Rows
        for i, row in csvData.iterrows():
            sql = "INSERT INTO data_base (train_X, train_Y, test_X, test_Y) VALUES (%s, %s, %s, %s)"
            value = (row['test_X'], row['test_Y'], row['train_X'], row['train_Y'])
            mycursor.execute(sql, value)

            mydb.commit()
        analyz(file_name)

    def analyz(file_name):
        train_val_data = parser_csv_file(os.path.join(r'src/static/files', file_name))

        print(file_name)
        # train_val_data = acc_data[file_name]

        acc_feature = list([])

        acc_feature.append([*feature_extraction.acc_all_features(train_val_data), *feature_extraction.f_features(train_val_data), 'null'])
        acc_colum_names = feature_extraction.acc_feature_names()
        df_acc = pd.DataFrame(np.array(acc_feature), columns=acc_colum_names)
        x_predict_acc = df_acc.drop('target', axis=1)
        with open('src/model/rf_clf_acc', 'rb') as f:
            rf_clf_acc = cPickle.load(f)
        result = rf_clf_acc.predict_proba(x_predict_acc)
        result = result.tolist()
        print(result, end="")
        print("   Predict - ", end="")
        unit_to_multiplier = {
            result.index(max(result)) == 0: 'Normal',
            result.index(max(result)) == 1: 'Overfitting',
            result.index(max(result)) == 2: 'Underfitting',

        }
        print(unit_to_multiplier[1])
        global finaly_result
        global unit_to_mult
        unit_to_mult = unit_to_multiplier[1]
        finaly_result = result
        return result



    def parser_csv_file(file_name):
        """
        function to extract training and testing data from csv files
        :param dir_name: string with the name of the directory that contains the csv files
        :return: dictionary - {name_file : [train_data, test_data]}
        """
        dict_data = {}

        csvfile = open(os.path.join(file_name))
        csvreader = csv.reader(csvfile)
        train_data = []
        test_data = []
        full_data = []
        for ind_row, row in enumerate(csvreader):
            # skip column names
            if ind_row < 2:
                continue
            # check for empty data
            if row[0] == '' or row[1] == '' or row[2] == '' or row[3] == '':
                break
            train_data.append(float(row[3]))
            test_data.append(float(row[1]))

        full_data.append(train_data)
        full_data.append(test_data)
        dict_data[file_name] = full_data

        return full_data


    return app
