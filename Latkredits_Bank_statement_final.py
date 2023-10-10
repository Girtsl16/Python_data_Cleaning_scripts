#importing necessary libraries
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from sqlalchemy.sql import text
predefined_length = 6

cols_names= ["payment_date", "Payment_type", "Client_Info", "blank", "Payment_nr", "Blank2", "Amount"] #column names for the dataframe

data = (
        pd.read_csv("C:/Users/GirtsL/Desktop/Python/LAT 4_10 aug (1).csv", #put in csv file name
                    header=None, #define that headers are not present in the csv file
                    names=cols_names, #headers are defined in the list above
                    skiprows=4, #skip the first 4 rows
                    sep='|', #define the separator
                    encoding='latin-1', #define the encoding
                    index_col=False) #define that the index is not present in the csv file
            .dropna(axis=1, how="all") #drop columns that are completely empty
        )



data = pd.concat([data])
# Filtering out the income only transactions
data_filtered = data[data['Payment_type'].str.contains("IEN", na=False)]
data_filtered['transaction_type'] = "Credit"

# Filtering out the debet only transactions
data_debet = data[data['Payment_type'].str.contains("IZE", na=False)]
data_debet['transaction_type'] = "Debet"

#replacing , with . in the Amount column so we can convert it to float

(data_filtered['Amount'].replace({',': '.'}, regex=True, inplace=True))

#creating new column with float values
data_filtered['amount_float']=data_filtered['Amount'].astype(float)

# defining function that identifies cent transactions in bank statement
def f2(row):
    if row['amount_float'] < 10:
        val = "less than 10"
    else:
        val = "valid"
    return val

#iniating the function to identify cent transactions and creating new column with the result
data_filtered['C'] = data_filtered.apply(f2, axis=1)

# filtering out the cent transactions and assigning them to new dataframe
data_filtered_cent = data_filtered[data_filtered['C'].str.contains("less than 10", na=False)]
data_filtered_cent['validation'] = data_filtered_cent['C']

data_filtered_cent['extracted_substring'] = data_filtered_cent['Client_Info'].str.extract(r'([^-]{%d}-[^-]{%d})' % (predefined_length, predefined_length))

# removing the " " from the extracted identity code
data_filtered_cent['identity_code']=data_filtered_cent['extracted_substring'].replace(' ','', regex=True)


# filtering out the valid transactions
data_filtered = data_filtered[data_filtered['C'].str.contains("valid", na=False)]


#defining length for identity code parts
predefined_length = 6

# extracting the identity code from the client info column using regex
data_filtered['extracted_substring'] = data_filtered['Client_Info'].str.extract(r'([^-]{%d}-[^-]{%d})' % (predefined_length, predefined_length))

# removing the " " from the extracted identity code
data_filtered['identity_code']=data_filtered['extracted_substring'].replace(' ','', regex=True)
# drop the extracted substring column
data_filtered.drop(['extracted_substring'], axis=1, inplace=True)
#defining length for bank account number
predefined_length_bank_account = 21

# Define a function to extract strings of desired length using regex
def extract_strings_with_length(row):
    text = row['Client_Info']
    pattern = r'\b\w{%d}\b' % predefined_length_bank_account
    matches = re.findall(pattern, text)
    return matches

# Apply the function to each row in the DataFrame
data_filtered['Bank_accountNR'] = data_filtered.apply(extract_strings_with_length, axis=1)

#Apply the function also for debet transactions
data_debet['Bank_accountNR'] = data_debet.apply(extract_strings_with_length, axis=1)
data_debet['Amount'] = data_debet['Blank2']

#defining function to extract agreement number from the client info column
def extract_seven_digit_strings(row):
    text = row['Client_Info']
    pattern = r'\b\d{7}\b'  # \b matches word boundaries, \d matches digits
    matches = re.findall(pattern, text)
    return matches
    # Use re.findall to find all matches in the input string
data_filtered['Test_agreementNR'] = data_filtered.apply(extract_seven_digit_strings, axis=1)

# creating new column with agreement number in string format so we can remove the [''] from the agreement number

data_filtered['agreementNR_str'] = data_filtered['Test_agreementNR'].astype(str)

#defining character that needs to replaced from agreementNR_str column and replacing it with empty string
replacements = {"'": '', "\]": '', "\[": ''}

#the use initiate these replacements on new column agreementNR
data_filtered['agreementNR']=data_filtered['agreementNR_str'].replace(replacements, regex=True)

#drop the agreementNR_str column
data_filtered.drop(['agreementNR_str'], axis=1, inplace=True)

#drop the Test_agreementNR column
data_filtered.drop(['Test_agreementNR'], axis=1, inplace=True)

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
    and broker_id in (11, 52) and broker_commission > 0
'''
with engine.connect() as conn:
    query = conn.execute(text(sql))   
df_sql = pd.DataFrame(query.fetchall())
data_filtered.drop(['blank'], axis=1, inplace=True)
data_filtered.drop(['Blank2'], axis=1, inplace=True)
# merge extracted data from the bank statement with the data from the database
df_sql = df_sql.astype(str)
df_merged = pd.merge(data_filtered, df_sql, left_on='identity_code', right_on='identity_code', how='inner')

# defining broker commission column as string so we can identify the rows that didn't joined because this identity code is not in the database
df_merged['broker_commission_str'] = df_merged['broker_commission'].astype(str)
#filtering out nan values from broker_commission_str column



# defining second SQL query to extract the data from the database and this time also extracting agreements, that have broker commission 0
sql_2 = '''
    SELECT * FROM dwh.bi_loan_agreement_py
    where sign_date <= (date_trunc('month', now()) + interval '1 month - 1 day')::date and sign_date > (date_trunc('month', now()) + interval '1 month - 1 day')::date - INTERVAL '4 months' -interval '5 day'
    and broker_id in (11, 52)
'''
with engine.connect() as conn:
    query = conn.execute(text(sql_2))   
df_sql_with_zero_broker_commission = pd.DataFrame(query.fetchall())

df_sql_with_zero_broker_commission = df_sql_with_zero_broker_commission.astype(str)
data_filtered = data_filtered.astype(str)

df_merged_2 = pd.merge(data_filtered, df_sql_with_zero_broker_commission, left_on='agreementNR', right_on='loan_response_id', how='inner')




df_merged = df_merged.astype(str)
df_merged_2 = df_merged_2.astype(str)

# concatinating both dataframes
df_merged_full = pd.concat([df_merged, df_merged_2])
df_merged_full.drop_duplicates(subset="loan_response_id", inplace=True)

#-----------------------------

df_merged_full['Amount'] = df_merged_full['Amount'].astype(float)
df_merged_full['broker_commission'] = df_merged_full['broker_commission'].astype(float)
# defining function that searches if the amount from the bank statement matches the broker commission in the database
def f(row):
    if row['Amount'] == row['broker_commission']:
        val = "Match Broker Commission"
    elif row['Amount'] == 0.01:
        val = "Cent Transfers"
    elif row['Amount'] > row['broker_commission']:
        val = "Bigger Than Broker Commission"
    elif row['Amount'] < row['broker_commission']:
        val = "Less Than Broker Commission"
    else:
        val = -1
    return val

# applying the function to each row in the dataframe



df_merged_full['Match'] = df_merged_full.apply(f, axis=1)

data_debet = data_debet.astype(str)
df_merged_full = df_merged_full.astype(str)


df_merged_full_with_debet = pd.merge(df_merged_full, data_debet, left_on='Bank_accountNR', right_on='Bank_accountNR', how='inner')

def f3(row):
    if row['Amount_x'] == row['Amount_y']:
        val = "Paid_back"
    else:
        val = "Not Paid Back"
    return val

#applying the function to each row in the df_merged_full_with_debet dataframe
df_merged_full_with_debet['validation'] = df_merged_full_with_debet.apply(f3, axis=1)

df_merged_full_with_debet['payment_date'] = df_merged_full_with_debet['payment_date_y']

df_merged_full_with_debet['Payment_type'] = df_merged_full_with_debet['Payment_type_y']

df_merged_full_with_debet['Client_Info'] = df_merged_full_with_debet['Client_Info_y']

df_merged_full_with_debet['transaction_type'] = df_merged_full_with_debet['transaction_type_y']

df_merged_full_with_debet['Amount'] = df_merged_full_with_debet['Amount_y']

df_merged_full_with_debet['Payment_nr'] = df_merged_full_with_debet['Payment_nr_y']

df_merged_full_new = pd.concat([df_merged_full,df_merged_full_with_debet])

df_merged_full_new = df_merged_full_new[df_merged_full_new.columns.drop(list(df_merged_full_new.filter(regex='_x')))]

df_merged_full_new = df_merged_full_new[df_merged_full_new.columns.drop(list(df_merged_full_new.filter(regex='_y')))]

df_merged_full_new.drop(['C'], axis=1, inplace=True)

data_filtered = data_filtered.loc[:, ['payment_date', 'Client_Info', 'Amount', 'transaction_type', 'Payment_nr']]

data_debet = data_debet.loc[:, ['payment_date', 'Client_Info', 'Amount', 'transaction_type', 'Payment_nr']]


data_debet = data_debet.astype(str)
data_filtered = data_filtered.astype(str)
data_filtered_cent = data_filtered_cent.astype(str)

df_merged_full['validation'] = df_merged_full['Match'].astype(str)

df_merged_full_new = pd.concat([df_merged_full,df_merged_full_with_debet])

df = df_merged_full_new.loc[:, ['payment_date', 'Client_Info', 'Amount', 'identity_code', 'transaction_type', 'validation', 'broker_id', 'application_id', 'loan_response_id', 'partner_id', 'Payment_nr']]


df_new = pd.concat([df, data_debet, data_filtered])

df_new.drop_duplicates(subset=['payment_date', 'Client_Info', 'Amount', 'transaction_type'], inplace=True)

df_new_full = pd.concat([df_new, data_filtered_cent])

df_new_full['payment_details'] = df_new_full['Client_Info']

df_new_full.drop(['Client_Info'], axis=1, inplace=True)

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


df = pd.DataFrame(df_new_full_2)

mark_rows_with_duplicates_and_any_paid_back(df, 'identity_code', ['validation', 'validation_2', 'validation_3'], 'Marked')

df_new_full_3 = pd.concat([df_new_full, df])

df_new_full_3.sort_values(by=['Marked'], ascending=False, inplace=True)
df_new_full_3=df_new_full_3.drop_duplicates(subset=['Payment_nr'])
df_new_full_3.to_excel('C:/Users/GirtsL/Desktop/Python/latkredits_september.xlsx')