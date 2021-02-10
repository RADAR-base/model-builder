import pandas as pd
from sqlalchemy import create_engine


def main(**args):
    user = args.get('user', 'postgres')
    password = args.get('password', "")
    df = pd.read_csv("data/wine-quality.csv")
    user_info = user + ":" + password
    engine = create_engine('postgresql://'+ user_info + '@localhost:5432/xyz')
    df.columns = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
           'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
           'pH', 'sulphates', 'alcohol', 'quality']
    df.to_sql('wine_dataset', engine, index=False)

if __name__ == "__main__":
    main()