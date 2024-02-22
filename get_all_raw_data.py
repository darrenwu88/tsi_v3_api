'''
Author: Darren Wu
Date: 12/5/2022


This script integrates all the base functions together, allowing the user to access all the raw data across all
their devices in the accounts in the form of a csv file output.


Checkout main() function for the main body of the script


PARAMS:
data_start_date: data start range
data_end_date: data end range
secrets_PATH: PATH for secrets.csv (all developer key/secret pairs)


WORKFLOW DOC: (will be updated as we make edits to the script)


Script job: Get, clean, and merge sensor data using the TSI v3 external API from specified dates.
Main inputs: developer key(s), secret(s), data date params (start_date & end_date)


### Most of the workflow can be explained through the main() function
1. Input developer key/secret pairs
2. Call client_token() function to return token .json file(s) by using developer key/secret pair
   until no key/secret pairs are left
3. Store token .json file(s) in ./client_tokens (file is named by developer email)
4. Open ./client_tokens folder and begin iterating through each token .json file in it.
5. For each token file, call device_list() which takes in the token .json file PATH as an argument.
   device_list() will extract the token string from the file input and GETS device list data associated with
   user token in json format.
   The json of device list data is then dumped and stored in ./device_list_by_developer_user.
   The list of device data (e.g. device_cloud_id, device_location, friendly_name, etc.) is also appended to a
   master .csv file using the append_device_list() function, which is found in ./master_device_list/master_device_list.csv.
   Within append_device_list(), we utilize shorten_name() and get_location() to obtain and serialize valuable data associated
   with each device, most notably location and a shortened friendly device ID.
6. Now, with the master .csv file of all the devices, we can call get_telemetry_flat() to request data for each device from
   the TSI server. The function (and GET method by extension) notably takes in the cloud ID of the device as an argument.
   get_telemetry_flat() returns a .json file that is stored in ./flat_telemetry_json_RAW. This function call is repeated
   for all devices located in master_device_list.csv
7. All of the .json data files in ./flat_telemetry_json_RAW is converted to .csv files and stored in ./flat_telemetry_csv_RAW
   using the flatten_json() function.
8. All of the .csv files in ./flat_telemetry_csv_RAW are merged together in one big .csv file. The merged file is outputted in
   the main directory as output_raw_telemetry.csv




OBJECTIVES for this week


include env + python interpreter documentation
#plan to add .yaml file to unite python env


assuming we still have the 30 day rule, script continuously appends data week by week
#potential solution could be uploading data to aws s3 buckets


'''
#debug error
import requests
import os
import json
import glob
import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np
import sys
from tkinter import *


# Note: Since this code uses the tkdatetimepicker package, make sure to install it by
# running 'pip install tkdatetimepicker' before running the code.
def get_dates():
    # Initialize default start and end dates
    thirty_days_ago = datetime.today() - timedelta(days=30)
    default_start_date = thirty_days_ago.strftime("%Y-%m-%dT00:00:00.000Z")
    default_end_date = datetime.today().strftime("%Y-%m-%dT00:00:00.000Z")
    data_start_date = f"&start_date={default_start_date}"
    data_end_date = f"&end_date={default_end_date}"

    # Function to handle user input and update start and end dates
    def update_dates():
        nonlocal data_start_date, data_end_date

        # Get input from entries
        start_date_input = start_date_entry.get()
        end_date_input = end_date_entry.get()

        # Update start date
        if start_date_input:
            try:
                start_date = datetime.strptime(start_date_input, '%m-%d-%Y')
                if start_date < thirty_days_ago:
                    start_date = thirty_days_ago
                    message_label.config(text="Start date can't be more than 30 days ago. Adjusted to 30 days ago.")
                data_start_date = "&start_date=" + start_date.strftime("%Y-%m-%dT00:00:00.000Z")
            except ValueError:
                message_label.config(text="Invalid start date. Using default start date.")

        # Update end date
        if end_date_input:
            try:
                end_date = datetime.strptime(end_date_input, '%m-%d-%Y')
                if end_date > datetime.now():
                    end_date = datetime.now()
                    message_label.config(text="End date is beyond today. Defaulting to today.")
                data_end_date = "&end_date=" + end_date.strftime("%Y-%m-%dT00:00:00.000Z")
            except ValueError:
                message_label.config(text="Invalid end date. Using default end date.")

        # Close GUI window
        root.destroy()

    # Create GUI window
    root = Tk()
    root.title("Date Input")
    root.geometry("300x200")

    # Start Date Entry
    Label(root, text="Start date (mm-dd-yyyy):").pack(pady=5)
    start_date_entry = Entry(root)
    start_date_entry.pack(pady=5)

    # End Date Entry
    Label(root, text="End date (mm-dd-yyyy):").pack(pady=5)
    end_date_entry = Entry(root)
    end_date_entry.pack(pady=5)

    # Submit Button
    Button(root, text="Submit", command=update_dates).pack(pady=10)

    # Message Label
    message_label = Label(root, text="")
    message_label.pack(pady=5)

    root.mainloop()

    return data_start_date, data_end_date


#PARAMS
secrets_PATH = r'./account_auth_info/secrets.csv'

def client_token(client_key, client_secret) -> None:


    headers = {
        'Accept': 'application/json',
    }
    params = {
        'grant_type': 'client_credentials',
    }
    data = {
        'client_id': client_key,
        'client_secret': client_secret,
    }
   
    response = requests.post(
        'https://api-prd.tsilink.com/api/v3/external/oauth/client_credential/accesstoken',
        params=params,
        headers=headers,
        data=data,
    )
   
    data = response.json()




    #create output token filename, format = '(DEVELOPER_EMAIL)_token.json'
    dev_email = data['developer.email']
    output_token_filename = f'({dev_email})_token.json'


    with open(os.path.join(r'./client_tokens', output_token_filename), 'w') as f:
        json.dump(data, f)


#shorten name based off de Foy's rules
def shorten_name(friendly_name, country_code, is_indoor_flag) -> str:
    #de Foy's code


    #determine if friendly_name has outdoor/indoor in its name (TBD)


    #pattern guide for str replacement
    pg = {'of': '_',
          'university': 'u',
          'univ': 'u',
          'airport:': 'apt',
          'campus': '',
          'school': '',
          ' f': '_',
          ',': '_',
          '.': '_',
          '-': '_',
          '/': '_',
          '(': '_',
          ')': '_',
          '__': '_'
          }


    #method for replacing string using pattern guide dict
    def str_replace(string, pattern_guide) -> str:
        for pattern in pattern_guide:
            string = string.replace(pattern, pattern_guide[pattern])


        return string


    #lowercase friendly name
    friendly_name = friendly_name.lower()
    #replace patterns in friendly_name according to pattern guide
    friendly_name = str_replace(friendly_name, pg)
    #add countrycode to beginning of short name
    country_code = country_code.lower()
    friendly_name = country_code + '_' + friendly_name
    #append is_indoors flag (type bool) to the short name
    friendly_name = friendly_name + '_' + str(is_indoor_flag)
   
    return friendly_name


#get country based of coords
def get_location(latitude, longitude) -> list:
    #search requires list input
    lat = latitude
    lon = longitude
   
    if (float(lat) != 0.0) and (float(lon) != 0.0):
        url = f'https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&accept-language=en&zoom=14'
        try:
            result = requests.get(url=url)
            result_json = result.json()
            # if state or city doesn't exist in json leave as null
            city = result_json['address']['city'] if 'city' in result_json['address'] else 'null'
            state = result_json['address']['state'] if 'state' in result_json['address'] else 'null'
            country = result_json['address']['country']
            country_code = result_json['address']['country_code']
            #too many keywords for diff countries when scope exceeds city-level
            full_address = result_json['address']
            return [city, state, country, country_code, full_address]
       
        except:
            return ['null', 'null', 'null', 'null', 'null']
   
    return ['null', 'null', 'null', 'null', 'null']


#append new countries to master device list
def append_device_list(response_json, dev_email) -> None:
    #open master_device_list in subdirectory
    df = pd.read_csv(os.path.join(r'./master_device_list', 'master_device_list.csv'))


    count = 0
    for device in response_json:
        cloud_account_id = device['account_id']
        cloud_device_id = device['device_id']
        serial_number = device['serial']
        friendly_name = device['metadata']['friendlyName']
        is_indoor = device['metadata']['is_indoor']
        longitude = device['metadata']['longitude']
        latitude = device['metadata']['latitude']
        #list inputs
        coords = (float(device['metadata']['latitude']), float(device['metadata']['longitude']))
        #prev coords
        prev_coords = (0.0, 0.0) #default val


        #debug print
        count = count + 1
        # TODO Add harmonization factor (user multi offset factor to this print statement)
        print(count, " ", dev_email, " ", serial_number, " ", friendly_name)


        #if device already exists in master list
        if cloud_device_id in df['cloud_device_id'].values:


            prev_device_row = df[df['cloud_device_id'] == cloud_device_id]
            #for some reason, pandas converts the coords tuple to string, eval converts the
            #stirng representation of the tuple back into an actual tuple
            prev_device_coords = eval(prev_device_row['coords'].values[0])


            #if new coords do not match existing coords (implying location change), update prev_coords and coords and location
            if coords != prev_device_coords:
                #update prev_coords with old lat/long coords
                prev_coords = prev_device_coords
                #drop the prev_device_row from df
                df = df[df['cloud_device_id'] != cloud_device_id]
            else:
                #location is still matching, no need to call get_location
                continue


        #get city/country/country_code based off coords
        #[city, state, country, country_code]
        address = get_location(latitude, longitude)  
        city = address[0]
        country = address[2]
        country_code = address[3]
        full_address = address[4]


        #convert coords to tuple type
        #coords = (longitude, latitude)
       
        #shorten friendly_name using shorten_name() subroutine
        short_name = shorten_name(friendly_name, country_code, is_indoor)


        #new device to be inserted into csv
        insert_row = {
            'cloud_account_id': cloud_account_id,
            'cloud_device_id': cloud_device_id,
            'serial': serial_number,
            'city': city,
            'country': country,
            'full_address': full_address,
            'friendly_name': friendly_name,
            'short_name': short_name,
            'dev_email': dev_email,
            'is_indoor': is_indoor,
            'coords': coords,
            'prev_coords': prev_coords
        }


        #check to see if coord has changed for existing sensors and overwrite, transfer previous coord location to prev_coords col
        #df.loc[(df['cloud_device_id'] == cloud_device_id) & (df['coords'] != coords), ['prev_coords']] = df['coords']
        #df.loc[(df['cloud_device_id'] == cloud_device_id) & (df['coords'] != coords), ['coords']] = coords
        #append new row for new device if it doesn't exist
        df = pd.concat([df, pd.DataFrame([insert_row])])


    #overwrite old csv file
    df.to_csv(os.path.join(r'./master_device_list', 'master_device_list.csv'), index = False)


def device_list(token_json_file) -> None:


    #get token from json file
    token_PATH = token_json_file
    json_file = open(token_PATH)
    #get data inside json file
    data = json.load(json_file)
   
    token = data['access_token']


    #request v3 data using token
    ### Note use of include_shared ("true" not recommended)
    requestUrl = "https://api-prd.tsilink.com/api/v3/external/devices/legacy-format?include_shared=false"
    requestHeaders = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/json"
    }


    response = requests.get(requestUrl, headers=requestHeaders)
    response_json = response.json()


   
    #get developer email for output file naming schema
    dev_email = data['developer.email']


    append_device_list(response_json, dev_email)


    #we want the output file to be a device list (identifiable by dev email), format = '(DEVELOPER EMAIL)_device_list.json'
    with open(os.path.join(r'./device_list_by_developer_user', f'{dev_email}_device_list.json'), "w") as outfile:
        outfile.write(response.text)

def get_and_flatten_telemetry(token_json_file, device_id, friendly_name, start_date, end_date, output_file, include_headers=True):
    # Initialize PATH where token for user is located
    token_path = f'./client_tokens/{token_json_file}'
    with open(token_path) as json_file:
        data = json.load(json_file)
    token = data['access_token']

    # Define telemetry query parameters
    telem_query = ['model', 'serial', 'location', 'is_public', 'is_indoor', 'mcpm1x0', 'mcpm2x5', 'mcpm4x0', 'mcpm10',
                   'ncpm0x5', 'ncpm1x0', 'ncpm2x5', 'ncpm4x0', 'ncpm10', 'tpsize', 'temperature', 'rh']

    # Construct the request URL
    request_url = f"https://api-prd.tsilink.com/api/v3/external/telemetry?device_id={device_id}&start_date={start_date}&end_date={end_date}"
    for arg in telem_query:
        request_url += '&telem[]=' + arg

    # Set the request headers
    request_headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}"
    }

    # Make the API request
    response = requests.get(request_url, headers=request_headers)

    # Check if the request was successful
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}, Device ID reference: {device_id}")

    # Process JSON data
    json_data = response.json()
    if not json_data:
        return  # Exit if json_data is empty

    # Define the specific measurement types you're expecting
    expected_measurements = ['timestamp', 'user_multi_factor', 'ncpm1x0', 'ncpm0x5', 'ncpm2x5', 'ncpm10', 'temp_c', 'mcpm2x5', 
                         'rh_percent', 'ncpm4x0', 'mcpm4x0', 'mcpm10', 'mcpm1x0', 'tpsize']

    all_rows = []
    for entry in json_data:
        base_data = {
            'model': entry.get('model'),
            'serial': entry.get('serial'),
            'cloud_account_id': entry.get('cloud_account_id'),
            'cloud_device_id': entry.get('cloud_device_id'),
            'is_indoor': entry['metadata'].get('is_indoor'),
            'is_public': entry['metadata'].get('is_public'),
            'latitude': entry['metadata']['location'].get('latitude'),
            'longitude': entry['metadata']['location'].get('longitude')
        }

        for sensor in entry.get('sensors', []):
            sensor_data = {
                'status_int': sensor.get('status_int'),
                'status': tuple(sensor.get('status', []))
            }

            measurement_data = {m: None for m in expected_measurements}  # Initialize with None
            for measurement in sensor.get('measurements', []):
                measurement_data['timestamp'] = measurement.get('data', {}).get('timestamp')
                measurement_data['user_multi_factor'] = measurement.get('user_multi_factor')
                measurement_type = measurement.get('type')
                if measurement_type in expected_measurements:
                    measurement_data[measurement_type] = measurement.get('data', {}).get('value')

            row = {**base_data, **sensor_data, **measurement_data}
            all_rows.append(row)

    # Convert to DataFrame
    transformed_df = pd.DataFrame(all_rows)
    transformed_df['timestamp'] = pd.to_datetime(transformed_df['timestamp'], utc=True)

    # Define columns to group by and aggregate
    groupby_columns = ['cloud_account_id', 'cloud_device_id', 'model', 'serial', 'is_indoor', 'is_public',
                       'latitude', 'longitude', 'timestamp']
    aggregated_df = transformed_df.groupby(groupby_columns).agg('first').reset_index()
   
    file_exists = os.path.isfile(output_file)
    include_headers = include_headers and not file_exists
    aggregated_df.to_csv(output_file, mode='a', index=False, header=include_headers)

def level_zero_raw(raw_df):
    #set data type for serial numbers
    raw_df['serial'] = raw_df['serial'].astype(np.int64)
    #set data type for timestamps
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'], format='%Y-%m-%d %H:%M:%S%z').dt.tz_localize(None)


    #get time_delta (since data capping is dependent on time_delta of data [time delta between measurements])
    #measure_count is frequency of measurements per hour
    measure_count = raw_df.groupby(['serial', raw_df['timestamp'].dt.year, raw_df['timestamp'].dt.month, raw_df['timestamp'].dt.day, raw_df['timestamp'].dt.hour])['serial'].transform(len)
    raw_df['time_delta'] = 60 / measure_count


    return raw_df

def convert_to_hourly_df(raw_df):
    qualitative_columns = ['serial', 'country', 'friendly_name', 'cloud_account_id', 
                            'cloud_device_id', 'is_indoor', 'is_public', 'latitude', 'longitude']
    agg_columns = ['PM10 (ug/m3)', 'PM1.0 (ug/m3)', 'PM2.5 (ug/m3)', 'PM4.0 (ug/m3)', 
                    'PM0.5 NC (#/cm3)', 'PM10 NC (#/cm3)', 'PM1.0 NC (#/cm3)', 'PM2.5 NC (#/cm3)', 
                    'PM4.0 NC (#/cm3)', 'Relative Humidity (%)', 'Temperature (Celsius)', 'Typical Particle Size (um)']

    # Common processing steps
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'], format='%Y-%m-%d %H:%M:%S').dt.tz_localize(None)
    raw_df['year'] = raw_df['timestamp'].dt.year
    raw_df['month'] = raw_df['timestamp'].dt.month
    raw_df['day'] = raw_df['timestamp'].dt.day
    raw_df['hour'] = raw_df['timestamp'].dt.hour
    grouped_key = ['serial', 'year', 'month', 'day', 'hour']

    raw_df['time_delta'] = raw_df.sort_values('timestamp').groupby('serial')['timestamp'].diff().fillna(method='bfill')
    raw_df['entry_count'] = raw_df.groupby(grouped_key)['serial'].transform('count')

    threshold_filter = lambda x: (len(x) * (x['time_delta'].value_counts(dropna=True).idxmax().total_seconds() / 60) >= 45)
    filtered_df = raw_df.groupby(grouped_key).filter(threshold_filter)

    grouped_hourly_df = filtered_df.groupby(grouped_key)[agg_columns].mean().round(2)
    grouped_hourly_df.reset_index(inplace=True)

    for col in ['year', 'month', 'day', 'hour']:
        grouped_hourly_df[col] = grouped_hourly_df[col].fillna(0).astype(int)

    grouped_hourly_df['timestamp'] = pd.to_datetime(grouped_hourly_df[['year', 'month', 'day', 'hour']])
    
    merge_df = raw_df[qualitative_columns + ['entry_count']].drop_duplicates()
    grouped_hourly_df = pd.merge(grouped_hourly_df, merge_df, on='serial', how='left')

    grouped_hourly_df.drop_duplicates(subset=['serial', 'timestamp'], inplace=True)
    grouped_hourly_df = grouped_hourly_df[['serial', 'timestamp'] + qualitative_columns[1:] + agg_columns + ['entry_count']]

    return grouped_hourly_df

def cap_data(df, column, time_cond, value_cond, cap_value):
    # Create a mask based on time and value conditions
    mask = (df['time_delta'] >= time_cond) & (df[column] >= value_cond)
   
    # Apply capping
    df.loc[mask, column] = cap_value
   
    # Update 'is_capped' column
    df.loc[mask, 'is_capped'] = True


def level_one_raw(lvl0_raw_df):

    ### First portion is to have case error filtering for the data
    # TODO V2 had case error filtering for the data, but v3 doesn't have 'Device Status' query parameter // will followup


    ### Second portion is to have data capping filters for the PM data
    #Data capping function for ug/m3 measurements
    # 1 min -> >5000 ug/m3
    # 5 min -> >2000 ug/m3
    # >10min -> >1000 ug/m3

    #get time_delta (since data capping is dependent on time_delta of data [time delta between measurements])
    #measure_count is frequency of measurements per hour
    measure_count = lvl0_raw_df.groupby(['serial', lvl0_raw_df['timestamp'].dt.year, lvl0_raw_df['timestamp'].dt.month, lvl0_raw_df['timestamp'].dt.day, lvl0_raw_df['timestamp'].dt.hour])['serial'].transform(len)
    lvl0_raw_df['time_delta'] = 60 / measure_count

    # Initialize the 'is_capped' column with False
    lvl0_raw_df['is_capped'] = False
   
    # Define capping conditions
    conditions = [
        {'column': 'PM1.0 (ug/m3)', 'time_cond': 10, 'value_cond': 1000, 'cap_value': 1000},
        {'column': 'PM1.0 (ug/m3)', 'time_cond': 5, 'value_cond': 2000, 'cap_value': 2000},
        {'column': 'PM1.0 (ug/m3)', 'time_cond': 1, 'value_cond': 5000, 'cap_value': 5000},
        {'column': 'PM2.5 (ug/m3)', 'time_cond': 10, 'value_cond': 1000, 'cap_value': 1000},
        {'column': 'PM2.5 (ug/m3)', 'time_cond': 5, 'value_cond': 2000, 'cap_value': 2000},
        {'column': 'PM2.5 (ug/m3)', 'time_cond': 1, 'value_cond': 5000, 'cap_value': 5000},
        {'column': 'PM4.0 (ug/m3)', 'time_cond': 10, 'value_cond': 1000, 'cap_value': 1000},
        {'column': 'PM4.0 (ug/m3)', 'time_cond': 5, 'value_cond': 2000, 'cap_value': 2000},
        {'column': 'PM4.0 (ug/m3)', 'time_cond': 1, 'value_cond': 5000, 'cap_value': 5000},
        {'column': 'PM10 (ug/m3)', 'time_cond': 10, 'value_cond': 1000, 'cap_value': 1000},
        {'column': 'PM10 (ug/m3)', 'time_cond': 5, 'value_cond': 2000, 'cap_value': 2000},
        {'column': 'PM10 (ug/m3)', 'time_cond': 1, 'value_cond': 5000, 'cap_value': 5000},
    ]
   
    # Apply capping based on conditions
    for cond in conditions:
        cap_data(lvl0_raw_df, **cond)

    return lvl0_raw_df

def level_two_raw(lvl1_raw_df):
    return

def delete_intermediate_files() -> None:
    delete_DIR_list = [r'device_list_by_developer_user',
                       r'flat_telemetry_json_RAW',
                       r'flat_telemetry_csv_RAW']
   
    for dir in delete_DIR_list:
        dir = os.path.join(dir, "*")
        files = glob.glob(dir)
        for file in files:
            #don't remove test files
            if 'test.txt' not in file:
                os.remove(file)

def main(secrets_PATH) -> None:
    # Retrieve user input for start and end dates
    start_date, end_date = get_dates()
    start_date = start_date.split('=')[1]
    end_date = end_date.split('=')[1]

    # Load API access secrets from CSV
    secrets_df = pd.read_csv(secrets_PATH)

    # Process each secret to obtain or refresh tokens
    for _, row in secrets_df.iterrows():
        client_key = row['key']
        client_secret = row['secret']
        dev_email = row['dev_email']

        # Token file name and path
        token_fname = f'({dev_email})_token.json'
        complete_path = os.path.join(r'./client_tokens', token_fname)
        tok_expires, BUFFER = 86400, 180  # Token expiration and buffer time

        # Check token validity and refresh if expired
        if os.path.exists(complete_path):
            modified = os.path.getmtime(complete_path)
            now = datetime.now().timestamp()
            if now - modified >= (tok_expires - BUFFER):
                print(f'Token expired: Refreshing token for {dev_email}')
                os.remove(complete_path)
                client_token(client_key, client_secret)
        else:
            print(f'No token found: Generating new token for {dev_email}')
            client_token(client_key, client_secret)
        print(f'Token for {dev_email} is ready')

    # Update device list from all tokens
    for token_json in glob.glob(os.path.join(r'./client_tokens', '*.json')):
        device_list(token_json)
    print('Device list is successfully updated.')

    # Process telemetry data for each device
    device_list_df = pd.read_csv(os.path.join(r'./master_device_list', 'master_device_list.csv'))
    # Single output file for all telemetry data
    output_csv_path = os.path.join(r'./telemetry_outputs', 'Raw_Edited.csv')

    # Ensure the file is empty
    if os.path.exists(output_csv_path):
        os.remove(output_csv_path)


    # Get and flatten telemetry for each possible device
    for _, row in device_list_df.iterrows():
        token_json_file = f'({row["dev_email"]})_token.json'
        get_and_flatten_telemetry(token_json_file, row['cloud_device_id'], row['short_name'], start_date, end_date, output_csv_path)

    # Now read the combined CSV for further processing
    combined_csv = pd.read_csv(output_csv_path)

    # Delete intermediate files if necessary
    delete_intermediate_files()
    combined_csv['timestamp'] = pd.to_datetime(combined_csv['timestamp']).dt.tz_localize(None)
    combined_csv.sort_values(by='timestamp', inplace=True)

    # Merge additional data from master device list
    master_device_list_df = pd.read_csv(os.path.join(r'./master_device_list', 'master_device_list.csv'))
    combined_csv = pd.merge(combined_csv, master_device_list_df[['serial', 'country', 'friendly_name']], on='serial', how='left')

    # Renaming and reordering columns as needed
    new_order = [
        'model', 'serial', 'timestamp', 'friendly_name', 'country',
        'cloud_account_id', 'cloud_device_id', 'is_indoor', 'is_public',
        'latitude', 'longitude', 'status', 'status_int', 'user_multi_factor', 'mcpm1x0', 'mcpm2x5',
        'mcpm4x0', 'mcpm10', 'ncpm0x5', 'ncpm1x0', 'ncpm2x5', 'ncpm4x0',
        'ncpm10', 'tpsize', 'temp_c', 'rh_percent'
    ]

    columns_mapping = {
        'mcpm1x0': 'PM1.0 (ug/m3)',
        'mcpm2x5': 'PM2.5 (ug/m3)',
        'mcpm4x0': 'PM4.0 (ug/m3)',
        'mcpm10': 'PM10 (ug/m3)',
        'ncpm0x5': 'PM0.5 NC (#/cm3)',
        'ncpm1x0': 'PM1.0 NC (#/cm3)',
        'ncpm2x5': 'PM2.5 NC (#/cm3)',
        'ncpm4x0': 'PM4.0 NC (#/cm3)',
        'ncpm10': 'PM10 NC (#/cm3)',
        'tpsize': 'Typical Particle Size (um)',
        'temp_c': 'Temperature (Celsius)',
        'rh_percent': 'Relative Humidity (%)'
    }
    combined_csv = combined_csv[new_order]
    combined_csv = combined_csv.rename(columns=columns_mapping)
    print('Merged raw csv successfully compiled.')

    # Save final telemetry data
    combined_csv.to_csv(os.path.join(r'./telemetry_outputs', 'Raw_Edited.csv'), index=False)

    lvl0_raw_df = level_zero_raw(combined_csv)
    lvl0_hourly_df = convert_to_hourly_df(lvl0_raw_df)


    lvl0_raw_df.to_csv(os.path.join(r'./telemetry_outputs', 'Level0.csv'),
                    index = False)

    lvl0_hourly_df.to_csv(os.path.join(r'./telemetry_outputs', 'Level0_hourly.csv'),
                        index = False)


    print('Level 0 QA completed')


    ### Level 1 QA
    lvl1_raw_df = level_one_raw(combined_csv)
    lvl1_hourly_df = convert_to_hourly_df(lvl1_raw_df)


    lvl1_raw_df.to_csv(os.path.join(r'./telemetry_outputs', 'Level1.csv'),
                    index = False)
                       
    lvl1_hourly_df.to_csv(os.path.join(r'./telemetry_outputs', 'Level1_hourly.csv'),
                        index = False)


    print('Level 1 QA completed')

    print('Data processing and quality assurance completed.')

if __name__ == "__main__":
    main(secrets_PATH)
