import os 
import glob


delete_DIR_list = [
                       r'device_list_by_developer_user', 
                       r'flat_telemetry_json_RAW', 
                       r'flat_telemetry_csv_RAW', ] 
    
for dir in delete_DIR_list:
    dir = os.path.join(dir, "*")
    files = glob.glob(dir)
    for file in files:
        if 'test.txt' not in file:
            os.remove(file)