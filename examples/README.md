# MLFlow Examples

In these examples, we have to fetch the data from the postgres dataset.

## Dataset

in this example, we have used Wine Quality dataset. You can download the wine quality dataset from [here](http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv). Now we have to upload it to postgres database.

### Creating Database in postgresql

Change the user to postgres :

```
sudo -u postgres psql
```

Create User for Postgres

```
$ create user <username>
```

Create Database

```
$ create DATABASE <databasename>
```

Provide the privileges to the postgres user

```
$ alter user <username> with encrypted password '<password>';
$ grant all privileges on database <database_name> to <username>;
```

### Uploading data to postgres database

Run the script in the example file:

```
python upload_data_to_postgres.py  --user=<username> --password=<password> --dbname=<database_name> --files data/wine-quality.csv --tablenames wine_dataset
```

This would create a table named `wine_dataset` in the database `<database_name>`.

You can check it by opening psql shell using:

`psql -U <username> -d <database_name>`

and then using `\d` in the shell.

## Running the Examples

To run the examples, you would need to go to `train.py` file in the example directory and add database details in line

`dbconn = PostgresPandasWrapper(dbname=<database_name>, user=<username>, password=<password>)`
