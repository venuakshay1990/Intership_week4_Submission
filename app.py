from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

with open('model_ICT_INTERNSHIP_007_decision_tree_02_21.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('x007_decision_tree_02_21.pkl', 'rb') as x:
    x = pickle.load(x)

sc = StandardScaler()
sc.fit(x)

app = Flask(__name__)

# Define the features to be log transformed
features_to_log_transform = ['Annual_Income', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate',
                             'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
                             'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age',
                             'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    Annual_Income = float(request.form['Annual_Income'])
    Num_Bank_Accounts = float(request.form['Num_Bank_Accounts'])
    Num_Credit_Card = float(request.form['Num_Credit_Card'])
    Interest_Rate = float(request.form['Interest_Rate'])
    Num_of_Loan = float(request.form['Num_of_Loan'])
    Delay_from_due_date = float(request.form['Delay_from_due_date'])
    Num_of_Delayed_Payment = float(request.form['Num_of_Delayed_Payment'])
    Changed_Credit_Limit = float(request.form['Changed_Credit_Limit'])
    Num_Credit_Inquiries = float(request.form['Num_Credit_Inquiries'])

    CreditMix = request.form['Credit_Mix']
    if CreditMix == "Bad":
        Credit_Mix = 1
    elif CreditMix == 'Standard':
        Credit_Mix = 2
    elif CreditMix == 'Good':
        Credit_Mix = 3

    Outstanding_Debt = float(request.form['Outstanding_Debt'])
    Credit_Utilization_Ratio = float(request.form['Credit_Utilization_Ratio'])
    Credit_History_Age = float(request.form['Credit_History_Age'])

    PaymentofMinAmount = request.form['Payment_of_Min_Amount']
    if PaymentofMinAmount == "Yes":
        Payment_of_Min_Amount = 1
    elif PaymentofMinAmount == 'No':
        Payment_of_Min_Amount = 2

    Total_EMI_per_month = float(request.form['Total_EMI_per_month'])
    Amount_invested_monthly = float(request.form['Amount_invested_monthly'])
    Monthly_Balance = float(request.form['Monthly_Balance'])

    Credit_Builder_Loan = request.form['Credit-Builder Loan']
    if Credit_Builder_Loan == "False":
        CBL = 0
    elif Credit_Builder_Loan == 'True':
        CBL = 1

    Personal_Loan = request.form['Personal Loan']
    if Personal_Loan == "False":
        PLL = 0
    elif Personal_Loan == 'True':
        PLL = 1

    Debt_Consolidation_Loan = request.form['Debt Consolidation Loan']
    if Debt_Consolidation_Loan == "False":
        DCL = 0
    elif Debt_Consolidation_Loan == 'True':
        DCL = 1

    Student_Loan = request.form['Student Loan']
    if Student_Loan == "False":
        STL = 0
    elif Student_Loan == 'True':
        STL = 1

    Payday_Loan = request.form['Payday Loan']
    if Payday_Loan == "False":
        PYL = 0
    elif Payday_Loan == 'True':
        PYL = 1

    Mortgage_Loan = request.form['Mortgage Loan']
    if Mortgage_Loan == "False":
        MEL = 0
    elif Mortgage_Loan == 'True':
        MEL = 1

    Auto_Loan = request.form['Auto Loan']
    if Auto_Loan == "False":
        AOL = 0
    elif Auto_Loan == 'True':
        AOL = 1

    Home_Equity_Loan = request.form['Home Equity Loan']
    if Home_Equity_Loan == "False":
        HEL = 0
    elif Home_Equity_Loan == 'True':
        HEL = 1

    Payment_Behaviour_High_spent_Large_value_payments = request.form['Payment_Behaviour_High_spent_Large_value_payments']
    if Payment_Behaviour_High_spent_Large_value_payments == "False":
        PBHLP = 0
    elif Payment_Behaviour_High_spent_Large_value_payments == 'True':
        PBHLP = 1

    Payment_Behaviour_High_spent_Medium_value_payments = request.form['Payment_Behaviour_High_spent_Medium_value_payments']
    if Payment_Behaviour_High_spent_Medium_value_payments == "False":
        PBHMP = 0
    elif Payment_Behaviour_High_spent_Medium_value_payments == 'True':
        PBHMP = 1

    Payment_Behaviour_High_spent_Small_value_payments = request.form['Payment_Behaviour_High_spent_Small_value_payments']
    if Payment_Behaviour_High_spent_Small_value_payments == "False":
        PBHSP = 0
    elif Payment_Behaviour_High_spent_Small_value_payments == 'True':
        PBHSP = 1

    Payment_Behaviour_Low_spent_Large_value_payments = request.form['Payment_Behaviour_Low_spent_Large_value_payments']
    if Payment_Behaviour_Low_spent_Large_value_payments == "False":
        PBLLP = 0
    elif Payment_Behaviour_Low_spent_Large_value_payments == 'True':
        PBLLP = 1

    Payment_Behaviour_Low_spent_Medium_value_payments = request.form['Payment_Behaviour_Low_spent_Medium_value_payments']
    if Payment_Behaviour_Low_spent_Medium_value_payments == "False":
        PBLMP = 0
    elif Payment_Behaviour_Low_spent_Medium_value_payments == 'True':
        PBLMP = 1

    Payment_Behaviour_Low_spent_Small_value_payments = request.form['Payment_Behaviour_Low_spent_Small_value_payments']
    if Payment_Behaviour_Low_spent_Small_value_payments == "False":
        PBLSP = 0
    elif Payment_Behaviour_Low_spent_Small_value_payments == 'True':
        PBLSP = 1

    feature = np.array([[Annual_Income, Num_Bank_Accounts, Num_Credit_Card, Interest_Rate, Num_of_Loan,
                         Delay_from_due_date, Num_of_Delayed_Payment, Changed_Credit_Limit, Num_Credit_Inquiries,
                         Credit_Mix, Outstanding_Debt, Credit_Utilization_Ratio, Credit_History_Age, Payment_of_Min_Amount,
                         Total_EMI_per_month, Amount_invested_monthly, Monthly_Balance, CBL, PLL, DCL, STL, PYL, MEL, AOL, HEL,
                         PBHLP, PBHMP, PBHSP, PBLLP, PBLMP, PBLSP]])

    # Convert the array to a DataFrame for easier manipulation
    feature_df = pd.DataFrame(feature, columns=[
        'Annual_Income', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate',
        'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
        'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt',
        'Credit_Utilization_Ratio', 'Credit_History_Age', 'Payment_of_Min_Amount',
        'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance', 'Credit-Builder Loan',
        'Personal Loan', 'Debt Consolidation Loan', 'Student Loan', 'Payday Loan',
        'Mortgage Loan', 'Auto Loan', 'Home Equity Loan', 'Payment_Behaviour_High_spent_Large_value_payments',
        'Payment_Behaviour_High_spent_Medium_value_payments', 'Payment_Behaviour_High_spent_Small_value_payments',
        'Payment_Behaviour_Low_spent_Large_value_payments', 'Payment_Behaviour_Low_spent_Medium_value_payments',
        'Payment_Behaviour_Low_spent_Small_value_payments'
    ])

    # Apply log transformation to specified features
    for feature in features_to_log_transform:
        feature_df[feature] = np.log1p(feature_df[feature])

    # Apply StandardScaler to the features
    feature_sc = sc.transform(feature_df)

    prediction = model.predict(feature_sc)

    return render_template('index.html',pred_res=prediction[0])


if __name__ == "__main__":
    app.run(debug=True)
