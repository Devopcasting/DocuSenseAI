from flask import Blueprint, render_template

# Create a blueprint
dashboard_route = Blueprint('dashboard', __name__, template_folder='templates')

@dashboard_route.route('/', methods=['GET', 'POST'])
def dashboard():
    return render_template('dashboard/dashboard.html', title="DocuSense AI: Home")