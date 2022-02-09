from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
app.run()


@app.route('/')
@app.route('/home')
def home_page():
    states = ['ALABAMA', 'ALASKA', 'ARIZONA', 'ARKANSAS', 'CALIFORNIA',
              'COLORADO', 'CONNECTICUT', 'DELAWARE', 'DISTRICT OF COLUMBIA',
              'FLORIDA', 'GEORGIA', 'HAWAII', 'IDAHO', 'ILLINOIS', 'INDIANA',
              'IOWA', 'KANSAS', 'KENTUCKY', 'LOUISIANA', 'MAINE', 'MARYLAND',
              'MASSACHUSETTS', 'MICHIGAN', 'MINNESOTA', 'MISSISSIPPI',
              'MISSOURI', 'MONTANA', 'NA', 'NEBRASKA', 'NEVADA', 'NEW HAMPSHIRE',
              'NEW JERSEY', 'NEW MEXICO', 'NEW YORK', 'NORTH CAROLINA',
              'NORTH DAKOTA', 'OHIO', 'OKLAHOMA', 'OREGON', 'PENNSYLVANIA',
              'PUERTO RICO', 'RHODE ISLAND', 'SOUTH CAROLINA', 'SOUTH DAKOTA',
              'TENNESSEE', 'TEXAS', 'UTAH', 'VERMONT', 'VIRGINIA', 'WASHINGTON',
              'WEST VIRGINIA', 'WISCONSIN', 'WYOMING']

    return render_template('index.html', states=states)


@app.route('/handle_data', methods=['POST'])
def handle_data():
    final_data = []

    state = request.form['state']
    emp = request.form['E']
    soc = request.form['SOC']
    jt = request.form['JT']
    income = request.form['income']

    df_h1b = joblib.load("df_h1b_copy.joblib")

    standard_states = ['ALABAMA', 'ALASKA', 'ARIZONA', 'ARKANSAS', 'CALIFORNIA',
                       'COLORADO', 'CONNECTICUT', 'DELAWARE', 'DISTRICT OF COLUMBIA',
                       'FLORIDA', 'GEORGIA', 'HAWAII', 'IDAHO', 'ILLINOIS', 'INDIANA',
                       'IOWA', 'KANSAS', 'KENTUCKY', 'LOUISIANA', 'MAINE', 'MARYLAND',
                       'MASSACHUSETTS', 'MICHIGAN', 'MINNESOTA', 'MISSISSIPPI',
                       'MISSOURI', 'MONTANA', 'NA', 'NEBRASKA', 'NEVADA', 'NEW HAMPSHIRE',
                       'NEW JERSEY', 'NEW MEXICO', 'NEW YORK', 'NORTH CAROLINA',
                       'NORTH DAKOTA', 'OHIO', 'OKLAHOMA', 'OREGON', 'PENNSYLVANIA',
                       'PUERTO RICO', 'RHODE ISLAND', 'SOUTH CAROLINA', 'SOUTH DAKOTA',
                       'TENNESSEE', 'TEXAS', 'UTAH', 'VERMONT', 'VIRGINIA', 'WASHINGTON',
                       'WEST VIRGINIA', 'WISCONSIN', 'WYOMING']
    df_state = {key: 0 for key in standard_states}
    df_state[state] = 1

    standard_emp = ['AR', 'HA', 'LA', 'MA', 'VHA', 'VLA']
    df_emp = {key: 0 for key in standard_emp}
    try:
        final_emp = df_h1b[df_h1b.EMPLOYER_NAME == emp.upper()].EMPLOYER_ACCEPTANCE.unique()[0]
        df_emp[final_emp] = 1
    except:
        df_emp['AR'] = 1

    standard_soc = ['AR', 'HA', 'LA', 'MA', 'VHA', 'VLA']
    df_soc = {key: 0 for key in standard_soc}
    try:
        final_soc = df_h1b[df_h1b.SOC_NAME == soc.upper()].SOC_ACCEPTANCE.unique()[0]
        df_soc[final_soc] = 1
    except:
        df_soc['AR'] = 1

    standard_jt = ['AR', 'HA', 'LA', 'MA', 'VHA', 'VLA']
    df_jt = {key: 0 for key in standard_jt}
    try:
        final_jt = df_h1b[df_h1b.JOB_TITLE == jt.upper()].JOB_ACCEPTANCE.unique()[0]
        df_jt[final_jt] = 1
    except:
        df_jt['AR'] = 1

    standard_income = ['HIGH', 'LOW', 'MEDIUM', 'VERY HIGH', 'VERY LOW']
    df_income = {key: 0 for key in standard_income}
    df_income[income] = 1

    final_data = list(df_state.values())[1:] + list(df_emp.values())[1:] + list(df_soc.values())[1:] + list(
        df_jt.values())[1:] + list(df_income.values())[1:]

    model = joblib.load("Gradient_Boosting.h5")

    return "Your Chances of H1-B Visa Approval is {}%".format(
        int(model.predict_proba(np.reshape(final_data, (-1, 71)))[:, 1] * 100))
