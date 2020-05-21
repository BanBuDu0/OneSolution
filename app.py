from flask import Flask, render_template, session, redirect, url_for, request, flash
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask_wtf.file import FileRequired, FileAllowed
from wtforms import StringField, SubmitField, FileField, SelectField
from wtforms.validators import DataRequired
import pandas as pd
import os

app = Flask(__name__)

app.config['SECRET_KEY'] = 'Author: SYJ'
app.config['UPLOAD_FOLDER'] = '//upload//'
bootstrap = Bootstrap(app)


class UpdateForm(FlaskForm):
    content = StringField('单个查询输入',
                          # validators=[DataRequired()]
                          )
    file = FileField('上传文件查询', validators=[
        FileRequired(),
        FileAllowed(['csv'], '只接收csv文件')
    ])
    submit = SubmitField('查询')


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UpdateForm()
    if form.validate_on_submit():
        try:
            upload_file = form.file.data
            upload_pd = pd.read_csv(upload_file)['ID']
            s = os.getcwd() + app.config['UPLOAD_FOLDER'] + upload_file.filename
            print(s)
            session['file_name'] = upload_file.filename
            upload_pd.to_csv(s, index=False)
            return redirect(url_for('res'))
        except KeyError as e:
            flash('Can not find index ' + e.__str__() + ' in CSV!')
    return render_template('index.html', form=form)


@app.route('/res')
def res():
    res_path = "D://jupyter_project//OneSolution//data//base_verify1.csv"
    res_file = pd.read_csv(res_path).fillna(0)
    t = {}
    for i, row in res_file.iterrows():
        t[int(row['ID'])] = int(row['TYPE'])
    r = []
    s = os.getcwd() + "//upload//" + session.get('file_name')
    uploaded_file = pd.read_csv(s)
    uploaded_file_numpy = uploaded_file.to_numpy()
    for i in uploaded_file_numpy:
        i = int(i)
        if i in t.keys():
            if t[i] == 1:
                r.append((i, '是'))
            else:
                r.append((i, '否'))
        else:
            r.append((i, '未知'))

    return render_template('res.html', row=r)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
