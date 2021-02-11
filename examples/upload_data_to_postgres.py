import pandas as pd
from sqlalchemy import create_engine
import argparse


def get_argparser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--user', action='store', type=str, help='username in psql databse',
                        default="postgres")
    parser.add_argument('--dbname', action='store', type=str, help='username in psql databse',
                        default="opendb")

    parser.add_argument('--password', action='store', type=str,
                        help='Password for the mention username', default="")
    return parser

def main():
    args = get_argparser().parse_args()
    user = args.user
    password = args.password
    dbname = args.dbname

    df = pd.read_csv("data/wine-quality.csv")
    user_info = user + ":" + password
    engine = create_engine('postgresql://'+ user_info + '@localhost:5432/' + dbname)
    df.columns = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
           'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
           'pH', 'sulphates', 'alcohol', 'quality']
    df.to_sql('wine_dataset', engine, index=False)

if __name__ == "__main__":
    main()