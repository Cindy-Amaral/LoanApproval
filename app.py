from dash import Dash, html, dcc, Input, Output, State, callback
import plotly.express as px
import pandas as pd
import pickle
import dash_bootstrap_components as dbc

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

with open('feature_encoder.pkl', 'rb') as file:
    feature_encoder = pickle.load(file)

with open('scalar.pkl', 'rb') as file:
    scalar = pickle.load(file)

with open('nb_model.pkl', 'rb') as file:
    nb = pickle.load(file)

gender_options = {'Female':'Female', 'Male':'Male'}
married_options = {'Yes':'Yes', 'No':'No'}
dependent_options = {0: '0',
                     1: '1',
                     2: '2',
                     3: '3+'
                     }
education_options = {'Graduate':'Graduate', 'Not Graduate':'Not Graduate'}
employed_options = {'Yes':'Yes', 'No':'No'}
credit_options = {1: 'Yes',
                  0: 'No'
                  }
area_options = {'Rural':'Rural', 'Urban':'Urban', 'Semiurban':'Semiurban'}

app.layout = html.Div(children=[
    html.H1(children='Loan Approval Dashboard'),
    html.Div(children='Select the options below to determine if you would be approved for a home loan.', style={'margin-bottom':'10px'}),

    html.Div(className='row', children=[
        html.Div(className='col-7', children=[
            html.Div(className='row', children=[
                html.Div(className='col-2', children=[
                    html.H6("Select Gender:", style={'fontSize':14})
                ]),
                html.Div(className='col-3', children=[
                    dcc.RadioItems(
                        id='gender_choice',
                        options = gender_options,
                        inline=True)
                ])
            ]),

            html.Div(className='row', children=[
                html.Div(className='col-2', children=[
                    html.H6("Married:", style={'fontSize':14})
                ]),
                html.Div(className='col-3', children=[
                    dcc.RadioItems(
                        id='married_choice',
                        options = married_options,
                        inline=True)
                ])
            ]),

            html.H6("Dependents:", style={'fontSize':14}),
            dcc.Dropdown(
                id='dependent_choice',
                options= dependent_options
            ),

            html.Div(className='row', children=[
                html.Div(className='col-2', children=[
                    html.H6("Education Level:", style={'fontSize':14})
                ]),
                html.Div(className='col-3', children=[
                    dcc.RadioItems(
                        id='education_choice',
                        options = education_options,
                        inline=True)
                ])
            ]),

            html.Div(className='row', children=[
                html.Div(className='col-2', children=[
                    html.H6("Self Employed:", style={'fontSize':14})
                ]),
                html.Div(className='col-3', children=[
                    dcc.RadioItems(
                        id='employed_choice',
                        options = employed_options,
                        inline=True)
                ])
            ]),

            html.H6("Applicant Income:", style={'fontSize':14}),
            dcc.Input(
                id='income',
                type='number',
                placeholder='applicant income'
            ),

            html.H6("Co-Applicant Income:", style={'fontSize':14}),
            dcc.Input(
                id='coincome',
                type='number',
                placeholder='coapplicant income'
            ),

            html.H6("Loan Amount:", style={'fontSize':14}),
            dcc.Input(
                id='loan_amount',
                type='number',
                placeholder='loan amount'
            ),

            html.H6("Loan Term (months):", style={'fontSize':14}),
            dcc.Slider(
                id='loan_term',
                min=12,
                max=360,
                step=12
            ),

            html.Div(className='row', children=[
                html.Div(className='col-2', children=[
                    html.H6("Credit History:", style={'fontSize':14})
                ]),
                html.Div(className='col-3', children=[
                    dcc.RadioItems(
                        id='credit_choice',
                        options = credit_options,
                        inline=True)
                ])
            ]),

            html.H6("Property Area:", style={'fontSize':14}),
            dcc.Dropdown(
                id='area_choice',
                options= area_options
            ),

            html.Button('Submit', id='submit_button'),
            html.Div(id='placeholder_for_output')
        ]),

        html.Div(className='col-5', children=[
                """test paragraph"""
        ])
    ])
], style = {'backgroundColor': colors['background'], 'color':colors['text'], 'margin':'0', 'padding':'0'})


@callback(
    Output('placeholder_for_output', 'children'),    
    Input('submit_button', 'n_clicks'),
    [
        State('gender_choice', 'value'),
        State('married_choice', 'value'),
        State('dependent_choice', 'value'),
        State('education_choice', 'value'),
        State('employed_choice', 'value'),
        State('income', 'value'),
        State('coincome', 'value'),
        State('loan_amount', 'value'),
        State('loan_term', 'value'),
        State('credit_choice', 'value'),
        State('area_choice', 'value')
    ],
    prevent_initial_call = True
)

def update_output(n_clicks, gender, married, dependents, education, employed, income, coincome, loan_amount, loan_term, credit, area):
    choice_list = [[gender, married, dependents, education, employed, income, coincome, loan_amount, loan_term, credit, area]]
    col_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed','ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term', 'Credit_History', 'Property_Area']
    df = pd.DataFrame(data=choice_list, columns=col_names)

    encoded = feature_encoder.transform(df[['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']])
    encoded_features = pd.DataFrame(encoded, columns=feature_encoder.get_feature_names_out())
    scaled = scalar.transform(df.drop(columns=['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents', 'Credit_History']))
    scaled_features = pd.DataFrame(scaled, columns=scalar.get_feature_names_out())
    X = pd.concat([df[['Dependents', 'Credit_History']], encoded_features, scaled_features], axis=1)
    
    prediction = nb.predict(X)
    if prediction == 1:
        pred_label = 'Approved'
    else:
        pred_label = 'Not Approved'
    
    return pred_label



if __name__ == "__main__":
    app.run(debug=True)