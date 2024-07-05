from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from wtforms import StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, ValidationError

# Upload PDF File Form
class UploadPDFForm(FlaskForm):
    file = FileField('PDF Document',validators=[DataRequired()])
    submit = SubmitField('Upload')

# Ask Question to PDF
class PDFQAForm(FlaskForm):
    get_answer_btn = SubmitField('Get Answer')
    question = StringField('Question', validators=[DataRequired()])
    answer =  TextAreaField('Answer')

