from flask import Flask, render_template, session, redirect, url_for, request
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask_wtf.file import FileRequired, FileAllowed
from wtforms import StringField, SubmitField, FileField, SelectField
from wtforms.validators import DataRequired
import pandas as pd
import os
import time

app = Flask(__name__)

app.config['SECRET_KEY'] = 'Author: SYJ'
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
        uploaded_file = pd.read_csv(form.file.data)
        test = uploaded_file.to_numpy()
        print(test[:10])
        return redirect(url_for('res'))
    return render_template('index.html', form=form)


@app.route('/res')
def res():
    return render_template('res.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
