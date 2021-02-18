import pandas as pd
from sqlalchemy import create_engine
import argparse
import os

def get_argparser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--user', action='store', type=str, help='username in psql databse',
                        default="postgres")
    parser.add_argument('--dbname', action='store', type=str, help='username in psql databse',
                        default="opendb")
    parser.add_argument('--password', action='store', type=str,
                        help='Password for the mention username', default="")
    parser.add_argument('-f','--files', nargs='+', help='csv files to upload', required=True)
    parser.add_argument('-t','--tablenames', nargs='+', help='tablename with respect to each file', required=True)
    return parser

def os_file_check(files):
    for file_location in files:
        if file_location.split('.')[-1] != "csv":
            raise FileNotFoundError(f"{file_location} is not csv file")
        if not os.path.isfile(file_location):
            raise FileNotFoundError(f"{file_location} does not exist!")

def main():
    args = get_argparser().parse_args()
    user = args.user
    password = args.password
    dbname = args.dbname
    files = args.files
    tablenames = args.tablenames
    if len(tablenames) != len(files):
        raise IOError("Number of input files must be equal to number of input tablenames")
    print(files)
    os_file_check(files)
    df_files = [pd.read_csv(filepath) for filepath in files]
    user_info = user + ":" + password
    for df in df_files:
        df.columns = [x.replace(" ", "_") for x in df.columns]

    engine = create_engine('postgresql://'+ user_info + '@localhost:5432/' + dbname)

    for i, df in enumerate(df_files):
        df.to_sql(tablenames[i], engine, index=False)

if __name__ == "__main__":
    main()
