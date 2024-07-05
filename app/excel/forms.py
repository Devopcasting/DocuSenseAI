from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from wtforms import StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, ValidationError

# Upload Excel File Form
class UploadExcelFForm(FlaskForm):
    file = FileField('PDF Document',validators=[DataRequired()])
    submit = SubmitField('Upload')

# Ask Question to Excel
class ExcelQAForm(FlaskForm):
    get_answer_btn = SubmitField('Get Answer')
    question = StringField('Question', validators=[DataRequired()])
    answer =  TextAreaField('Answer')

