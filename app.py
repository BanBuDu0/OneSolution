from flask import Flask, render_template, session, redirect, url_for, flash
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask_wtf.file import FileRequired, FileAllowed
from wtforms import StringField, SubmitField, FileField
from wtforms.validators import DataRequired, ValidationError
import pandas as pd
import os

app = Flask(__name__)

app.config['SECRET_KEY'] = 'Author: SYJ'
app.config['UPLOAD_FOLDER'] = '//upload//'
bootstrap = Bootstrap(app)


def validate_answer(form, field):
    s = pd.read_csv(field.data).columns
    print("ok")
    if len(s) != 1 or s[0] != 'ID':
        print("ok2")
        raise ValidationError('Can not find column \'ID\' in CSV!')


class TextForm(FlaskForm):
    content = StringField('单个查询输入',
                          validators=[DataRequired()]
                          )
    submit = SubmitField('查询')


class FileForm(FlaskForm):
    file = FileField('上传文件查询', validators=[
        FileRequired(),
        validate_answer,
        FileAllowed(['csv'], '只接收csv文件')
    ])

    submit = SubmitField('查询')


@app.route('/', methods=['GET', 'POST'])
def index():
    text_form = TextForm()
    file_form = FileForm()
    if file_form.validate_on_submit():
        upload_file = file_form.file.data
        upload_pd = pd.read_csv(upload_file)['ID']
        s = os.getcwd() + app.config['UPLOAD_FOLDER'] + upload_file.filename
        print(s)
        session['file_name'] = upload_file.filename
        session['method'] = 0
        upload_pd.to_csv(s, index=False)
        return redirect(url_for('res'))

    elif text_form.validate_on_submit():
        _input = text_form.content.data
        try:
            session['id'] = int(_input)
            session['method'] = 1
            return redirect(url_for('res'))
        except:
            flash('Not correct single integer ID')
    return render_template('index.html', text_form=text_form, file_form=file_form)


@app.route('/res')
def res():
    res_path = "D://jupyter_project//OneSolution//data//res.csv"
    res_file = pd.read_csv(res_path, encoding='gbk')
    res_file.columns = ['ID', 'TYPE']
    t = {}
    for i, row in res_file.iterrows():
        t[int(row['ID'])] = int(row['TYPE'])
    r = []
    s = os.getcwd() + "//upload//" + session.get('file_name')
    _method = session.get('method')
    if _method == 0:
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
    elif _method == 1:
        _id = session.get('id')
        if _id in t.keys():
            if t[_id] == 1:
                r.append((_id, '是'))
            else:
                r.append((_id, '否'))
        else:
            r.append((_id, '未知'))
    return render_template('res.html', row=r)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
