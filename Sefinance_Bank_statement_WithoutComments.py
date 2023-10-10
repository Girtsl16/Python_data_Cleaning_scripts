import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from sqlalchemy.sql import text

#defining column names for dataframes
cols_names= ["Bank_acount", "Payment_group", "payment_date", "Client_Info", "payment_details", "amount", "Currency", "Payment_Type", "date_indicator"]

data_3 = (
        pd.read_csv("SEF 1_3 sep.csv",
                    header=None,
                    names=cols_names,
                    sep=';',
                    encoding='latin-1',
                    index_col=False)
            .dropna(axis=1, how="all")
        )

data_2 = (
        pd.read_csv("SEF 4_10 sep.csv",
                    header=None,
                    names=cols_names,
                    sep=';',
                    encoding='latin-1',
                    index_col=False)
            .dropna(axis=1, how="all")
        )


data_1 = (
        pd.read_csv("SEF 11_17 sep.csv",
                    header=None,
                    names=cols_names,
                    sep=';',
                    encoding='latin-1',
                    index_col=False)
            .dropna(axis=1, how="all")
        )
# Merging all dataframes into one
data = pd.concat([data_1, data_2, data_3])

#setting random seed this is necessary so random numbers are reproducible
np.random.seed(42)
#creating new column with random numbers that will serve as the unique index of the dataframe
data['randomNumCol'] = np.random.randint(1,10000, size=len(data))


#Filtering out outgoing payments
data_Debet=data[(data.Payment_group == 20) & (data.Payment_Type == "D")]
#Filtering out incoming payments
data_Credit=data[(data.Payment_group == 20) & (data.Payment_Type == "K")]

data_Credit['transaction_type'] = "Credit"

data_Debet['transaction_type'] = "Debet"
data_Credit['source_column'] = "data_Credit"
data_Debet['source_column'] = "data_Debet"

#replacing , with . in the Amount column so we can convert it to float

(data_Credit['amount'].replace({',': '.'}, regex=True, inplace=True))

#creating new column with float values
data_Credit['amount_float']=data_Credit['amount'].astype(float)

# defining function that identifies cent transactions in bank statement
def Cent_transfers(row):
    if row['amount_float'] < 10:
        val = "less than 10"
    else:
        val = "valid"
    return val

#iniating the function to identify cent transactions and creating new column with the result
data_Credit['validation'] = data_Credit.apply(Cent_transfers, axis=1)

# filtering out the cent transactions and assigning them to new dataframe and creating new column that will define this dataframe source
data_Credit_cent = data_Credit[data_Credit['validation'].str.contains("less than 10", na=False)]
data_Credit_cent['source_column'] = "data_Credit_cent"



# filtering out the valid transactions



predefined_length = 6

 
data_Credit['extracted_identity_code'] = data_Credit['Client_Info'].str.extract(r'([^-]{%d}-[^-]{%d})' % (predefined_length, predefined_length))
data_Credit_cent['extracted_identity_code'] = data_Credit_cent['Client_Info'].str.extract(r'([^-]{%d}-[^-]{%d})' % (predefined_length, predefined_length))
data_Credit_cent['identity_code']=data_Credit_cent['extracted_identity_code'].replace(' ','', regex=True)
data_Credit['identity_code']=data_Credit['extracted_identity_code'].replace(' ','', regex=True)
data_Credit.drop(['extracted_identity_code'], axis=1, inplace=True)
data_Credit_cent.drop(['extracted_identity_code'], axis=1, inplace=True)

    
length_of_bank_account_nr = 21

def extracting_bank_account_nr(row):
    text = row['Client_Info']
    pattern = r'\b\w{%d}\b' % length_of_bank_account_nr
    matches = re.findall(pattern, text)
    return matches

data_Credit['Bank_accountNR'] = data_Credit.apply(extracting_bank_account_nr, axis=1)
data_Debet['Bank_accountNR'] = data_Debet.apply(extracting_bank_account_nr, axis=1)


def extract_agreement_Nr(row):
    text = row['payment_details']
    pattern = r'\b\d{7}\b'  # \b matches word boundaries, \d matches digits
    matches = re.findall(pattern, text)
    return matches
 
data_Credit['Test_agreementNR'] = data_Credit.apply(extract_agreement_Nr, axis=1)



data_Credit['agreementNR_str'] = data_Credit['Test_agreementNR'].astype(str)


replacements = {"'": '', "\]": '', "\[": ''}


data_Credit['agreementNR']=data_Credit['agreementNR_str'].replace(replacements, regex=True)


data_Credit.drop(['agreementNR_str'], axis=1, inplace=True)


data_Credit.drop(['Test_agreementNR'], axis=1, inplace=True)

# DEFINE THE DATABASE CREDENTIALS (this is necessary because we need to compare data that is extracted from bank statement with the data in the database)
user = 'girtsliepins'
password = 'gohnhcwCp072U4Biqibt'
host = '159.148.40.86'
port = 5432
database = 'DWH'
 
# PYTHON FUNCTION TO CONNECT TO THE POSTGRESQL DATABASE AND
# RETURN THE SQLACHEMY ENGINE OBJECT
def get_connection():
    return create_engine(
        url="postgresql://{0}:{1}@{2}:{3}/{4}".format(
            user, password, host, port, database
        )
    )
 
 
if __name__ == '__main__':
 
    try:
        # GET THE CONNECTION OBJECT (ENGINE) FOR THE DATABASE
        engine = get_connection()
        print(
            f"Connection to the {host} for user {user} created successfully.")
    except Exception as ex:
        print("Connection could not be made due to the following error: \n", ex)

# defining SQL query to extract the data from the database
sql = '''
    SELECT * FROM dwh.bi_loan_agreement_py
    where sign_date <= (date_trunc('month', now()) + interval '1 month - 1 day')::date and sign_date > (date_trunc('month', now()) + interval '1 month - 1 day')::date - INTERVAL '4 months' -interval '5 day'
    and broker_id in (1) 
'''
with engine.connect() as conn:
    query = conn.execute(text(sql))   
df_sql_with_zero_broker_commission = pd.DataFrame(query.fetchall())
# merge extracted data from the bank statement with the data from the database

df_sql_with_zero_broker_commission = df_sql_with_zero_broker_commission.astype(str)
df_merged = pd.merge(data_Credit, df_sql_with_zero_broker_commission, left_on='agreementNR', right_on='loan_response_id', how='inner')

# defining broker commission column as string so we can identify the rows that didn't joined because this identity code is not in the database
df_merged['broker_commission_str'] = df_merged['broker_commission'].astype(str)
#filtering out nan values from broker_commission_str column

df_merged['source_column'] = "df_merged"


# defining second SQL query to extract the data from the database and this time also extracting agreements, that have broker commission 0
sql_2 = '''
    SELECT * FROM dwh.bi_loan_agreement_py
    where sign_date <= (date_trunc('month', now()) + interval '1 month - 1 day')::date and sign_date > (date_trunc('month', now()) + interval '1 month - 1 day')::date - INTERVAL '4 months' -interval '5 day'
    and broker_id in (1) and broker_commission > 0
'''
with engine.connect() as conn:
    query = conn.execute(text(sql_2))   
df_sql = pd.DataFrame(query.fetchall())

df_sql = df_sql.astype(str)
data_Credit = data_Credit.astype(str)

df_merged_2 = pd.merge(data_Credit, df_sql, left_on='identity_code', right_on='loan_response_id', how='inner')




df_merged = df_merged.astype(str)
df_merged_2 = df_merged_2.astype(str)
df_merged_2['source_column'] = "df_merged_2"
# concatinating both dataframes
df_merged_full = pd.concat([df_merged, df_merged_2])
df_merged_full.drop_duplicates(subset="loan_response_id", inplace=True)
df_merged_full['source_column'] = 'df_merged_full'
#-----------------------------

df_merged_full['amount'] = df_merged_full['amount'].astype(float)
df_merged_full['broker_commission'] = df_merged_full['broker_commission'].astype(float)
df_merged_full['monthly_payment'] = df_merged_full['monthly_payment'].astype(float)
df_merged_full['contract_fee'] = df_merged_full['contract_fee'].astype(float)
# defining function that searches if the amount from the bank statement matches the broker commission or monthly payment or registration fee in the database
def validation(row):
    if row['amount'] == row['broker_commission']:
        val = "Match Broker Commission"
    elif row['amount'] == 0.01:
        val = "Cent Transfers"
    elif row['amount'] == row['monthly_payment']:
        val = "Matches Monthly Payment"
    elif row['amount'] == row['contract_fee']:
        val = "Matches Registration Fee"
    elif row['amount'] < row['broker_commission']:
        val = "Less Than Broker Commission"
    elif row['amount'] > row['broker_commission']:
        val = "Bigger Than Broker Commission"
    else:
        val = -1
    return val



# applying the function to each row in the dataframe



df_merged_full['validation'] = df_merged_full.apply(validation, axis=1)

data_Debet = data_Debet.astype(str)
df_merged_full = df_merged_full.astype(str)


df_merged_full_with_Debet = pd.merge(df_merged_full, data_Debet, left_on='Bank_accountNR', right_on='Bank_accountNR', how='inner')
(df_merged_full_with_Debet['amount_y'].replace({',': '.'}, regex=True, inplace=True))
(df_merged_full_with_Debet['amount_x'].replace({',': '.'}, regex=True, inplace=True))
df_merged_full_with_Debet['amount_x'] = df_merged_full_with_Debet['amount_x'].astype(float)
df_merged_full_with_Debet['amount_y'] = df_merged_full_with_Debet['amount_y'].astype(float)


def validation_Paid_Back(row):
    if row['amount_x'] == row['amount_y']:
        val = "Paid_back"
    else:
        val = "Not Paid Back"
    return val

#applying the function to each row in the df_merged_full_with_Debet dataframe
df_merged_full_with_Debet['validation'] = df_merged_full_with_Debet.apply(validation_Paid_Back, axis=1)

#applying the function to each row in the df_merged_full_with_Debet dataframe
df_merged_full_with_Debet['validation'] = df_merged_full_with_Debet.apply(validation_Paid_Back, axis=1)


df_merged_full_with_Debet['payment_date'] = df_merged_full_with_Debet['payment_date_y']

df_merged_full_with_Debet['Payment_Type'] = df_merged_full_with_Debet['Payment_Type_y']

df_merged_full_with_Debet['Client_Info'] = df_merged_full_with_Debet['Client_Info_y']

df_merged_full_with_Debet['transaction_type'] = df_merged_full_with_Debet['transaction_type_y']

df_merged_full_with_Debet['amount'] = df_merged_full_with_Debet['amount_y']

df_merged_full_with_Debet['payment_details'] = df_merged_full_with_Debet['payment_details_y']

df_merged_full_with_Debet['date_indicator'] = df_merged_full_with_Debet['date_indicator_y']


df_merged_full_with_Debet['randomNumCol'] = df_merged_full_with_Debet['randomNumCol_y']


df_merged_full_with_Debet['source_column'] = "df_merged_full_with_Debet"

df_merged_full_new = pd.concat([df_merged_full,df_merged_full_with_Debet])

df_merged_full_new = df_merged_full_new[df_merged_full_new.columns.drop(list(df_merged_full_new.filter(regex='_x')))]

df_merged_full_new = df_merged_full_new[df_merged_full_new.columns.drop(list(df_merged_full_new.filter(regex='_y')))]


df_merged_full_new['source_column'] = "df_merged_full_new"

data_Credit = data_Credit.loc[:, ['payment_date', 'Client_Info', 'amount', 'transaction_type', 'payment_details', 'Bank_accountNR', 'date_indicator', 'randomNumCol', 'source_column']]

data_Debet = data_Debet.loc[:, ['payment_date', 'Client_Info', 'amount', 'transaction_type', 'payment_details', 'Bank_accountNR', 'date_indicator', 'randomNumCol', 'source_column']]


data_Debet = data_Debet.astype(str)
data_Credit = data_Credit.astype(str)
data_Credit_cent = data_Credit_cent.astype(str)

df_merged_full['validation'] = df_merged_full['validation'].astype(str)

df_merged_full_new = pd.concat([df_merged_full,df_merged_full_with_Debet])

df_merged_full_new['extracted_identity_code'] = df_merged_full_new['Client_Info'].str.extract(r'([^-]{%d}-[^-]{%d})' % (predefined_length, predefined_length))


df_merged_full_new['identity_code']=df_merged_full_new['extracted_identity_code'].replace(' ','', regex=True)


df = df_merged_full_new.loc[:, ['payment_date', 'Client_Info', 'amount', 'identity_code', 'transaction_type', 'validation', 'broker_id', 'application_id', 'loan_response_id', 'partner_id', 'payment_details', 'Bank_accountNR', 'date_indicator', 'randomNumCol', 'source_column']]


df_new = pd.concat([df, data_Debet, data_Credit])


df_new_full = pd.concat([df_new, data_Credit_cent])



df_new_full['Marked'] = False

df_new_full_2 = df_new_full[(df_new_full.application_id != 'NaN')]

df_new_full_2 = df_new_full.replace('', np.nan, regex=False)

validation_filter = ['Match Broker Commission','Less Than Broker Commission','Bigger Than Broker Commission', 'Paid_back']

df_new_full['validation_2'] = df_new_full['validation']


df_new_full['validation_3'] = df_new_full['validation']

df_new_full_2 = df_new_full[df_new_full['validation'].isin(validation_filter)]



def mark_rows_with_duplicates_and_any_paid_back(df, id_column_name, validation_columns, marker_column_name):
    """
    Marks rows in a DataFrame with 'True' if there are duplicates in the identity code column and 
    at least one occurrence of 'Paid Back' is found in the validation columns for any row with the same identity code; 
    otherwise, it marks the row with 'False'.

    Parameters:
        df (pd.DataFrame): The DataFrame to be checked and marked.
        id_column_name (str): The name of the column containing identity codes.
        validation_columns (list): A list of column names to check for 'Paid_back'.
        marker_column_name (str): The name of the column where the 'True' or 'False' markers will be placed.

    Returns:
        None: The function modifies the input DataFrame in place.
    """
    # Create a boolean mask for rows that meet the conditions
    duplicate_mask = df[id_column_name].duplicated(keep=False)
    validation_mask = df[validation_columns].apply(lambda col: col == 'Paid_back' if col.dtype == 'O' else False, axis=1)
    combined_mask = duplicate_mask & validation_mask.any(axis=1)
    
    # Mark the rows with 'True' or 'False' based on the mask
    df[marker_column_name] = combined_mask
    
    # Find and mark other rows with the same identity code that meet the conditions
    df.loc[df.groupby(id_column_name)[marker_column_name].transform('any'), marker_column_name] = True

# Example usage:


df = pd.DataFrame(df_new_full_2)

df['Bank_accountNR'] = df.apply(extracting_bank_account_nr, axis=1)

df['Bank_accountNR'] = df['Bank_accountNR'].astype(str)

mark_rows_with_duplicates_and_any_paid_back(df, 'Bank_accountNR', ['validation', 'validation_2', 'validation_3'], 'Marked')

df_new_full_5 = pd.concat([df_new_full, df])

df_new_full_5.sort_values(by=['Marked'], ascending=False, inplace=True)

df_new_full_5['amount'] = df_new_full_5['amount'].astype(str)

df_new_full_5['randomNumCol']= df_new_full_5['randomNumCol'].astype(int)


df_new_full_5.drop_duplicates(subset=['randomNumCol', 'payment_details'], inplace=True, keep='first')