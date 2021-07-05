from model_class import  ModelClass
import pandas as pd
from datetime import  timedelta
import datetime
import numpy as np

class LungStudy(ModelClass):

    def __init__(self):
        super().__init__()
        self.window_size = 5
        self.query_heart = '''SELECT "PROJECTID"  as pid, \
            "USERID" as uid, \
            "SOURCEID"  as sid,\
            to_timestamp("TIME"  / 1000 )   as time,\
            extract(hour from to_timestamp("TIME" / 1000)) as hour,\
            date(to_timestamp("TIME"  / 1000 ) ) as date,\
            to_timestamp("WINDOW_START" / 1000 ) as window_start,\
            to_timestamp("WINDOW_END" / 1000 )  as window_end,\
            "COUNT" as heartrate_count,\
            "MIN" as heartrate_min,\
            "MAX" as heartrate_max,\
            "MEAN" as heartrate_mean,\
            "STDDEV" as heartrate_std,\
            "MEDIAN" as heartrate_median,\
            "MODE"  as heartrate_model,\
            "IQR" as heartrate_iqr,\
            "SKEW" as heartrate_skew \
            from "PUSH_GARMIN_HEART_RATE_SAMPLE_TIMESTAMP_LONG_WINDOWED_1H_TABLE"'''

        self.query_body_battery = '''SELECT "PROJECTID"  as pid, \
            "USERID" as uid, \
            "SOURCEID"  as sid, \
            to_timestamp("TIME"  / 1000 ) as time, \
            extract(hour from to_timestamp("TIME" / 1000)) as hour,\
            date(to_timestamp("TIME"  / 1000 ) ) as date,\
            to_timestamp("WINDOW_START" / 1000 ) as window_start,\
            to_timestamp("WINDOW_END" / 1000 )  as window_end,
            "COUNT" as body_battery_count, \
            "MIN" as body_battery_min, \
            "MAX" as body_battery_max, \
            "MEAN" as body_battery_mean, \
            "STDDEV" as body_battery_std, \
            "MEDIAN" as body_battery_median, \
            "MODE"  as body_battery_model, \
            "IQR" as body_battery_iqr, \
            "SKEW" as body_battery_skew \
            from "PUSH_GARMIN_BODY_BATTERY_TIMESTAMP_LONG_WINDOWED_1H_TABLE"'''

        self.query_pulse =  '''SELECT "PROJECTID"  as pid, \
            "USERID" as uid, \
            "SOURCEID"  as sid, \
            to_timestamp("TIME"  / 1000 )  as time, \
            extract(hour from to_timestamp("TIME" / 1000)) as hour,\
            date(to_timestamp("TIME"  / 1000 ) ) as date,\
            to_timestamp("WINDOW_START" / 1000 ) as window_start,\
            to_timestamp("WINDOW_END" / 1000 )  as window_end,
            "COUNT" as pulse_count, \
            "MIN" as pulse_min, \
            "MAX" as pulse_max, \
            "MEAN" as pulse_mean, \
            "STDDEV" as pulse_std, \
            "MEDIAN" as pulse_median, \
            "MODE"  as pulse_model, \
            "IQR" as pulse_iqr, \
            "SKEW" as pulse_skew \
            from "PUSH_GARMIN_PULSE_OX_TIMESTAMP_LONG_WINDOWED_1H_TABLE"'''

        self.query_activity = '''SELECT "projectId"  as pid, \
            "userId" as uid, \
            "sourceId"  as sid, \
            "time"  as time, \
            date("time") as date,\
            extract(hour from "time") as hour, \
            duration, distance, "activeKilocalories", "averageSpeed", steps \
            from push_garmin_activity_summary'''

        self.grouped_activity = f'''SELECT uid, date, hour,\
            sum(duration) as activity_duration, sum(distance) as activity_distance, \
            sum("activeKilocalories") as activity_calories, sum(steps) \
            as activity_steps from ({self.query_activity}) as query_activity \
            group by uid, date, hour'''

        self.sleep_activity = f'''SELECT \
            "userId" as uid, \
            "projectId" as pid, \
            "sleepLevel" as sleep_level, \
            to_timestamp("startTime") as sleep_start_time, \
            to_timestamp("endTime") as sleep_end_time, \
            time as time\
            from "push_garmin_sleep_level" as sleep_activity'''

        self.cat_score_retrieve_query = f'''SELECT "PROJECTID"  as pid, \
            "USERID" as uid, \
            "SOURCEID"  as sid, \
            to_timestamp("TIME"  / 1000 ) as time, \
            extract(hour from to_timestamp("TIME" / 1000)) as hour,\
            date(to_timestamp("TIME"  / 1000 ) ) as date,\
            "CAT_SCORE" as cat_score \
            from "QUESTIONNAIRE_CAT_SCORE_STREAM"'''


    def get_query_for_training(self):
        final_query =f'''SELECT * FROM ( {self.query_heart}  )  AS query_heart \
                        NATURAL JOIN ( {self.query_body_battery} ) AS query_body_battery \
                        NATURAL JOIN ( {self.query_pulse} ) AS query_pulse \
                        NATURAL LEFT JOIN ( {self.grouped_activity}) as activity'''

        return [final_query, self.sleep_activity, self.cat_score_retrieve_query]


    def get_query_for_prediction(self, user_id, starttime, endtime):

        if starttime is None and endtime is None:
            final_query =f'''SELECT * FROM ( {self.query_heart}  )  AS query_heart \
                            NATURAL JOIN ( {self.query_body_battery} ) AS query_body_battery \
                            NATURAL JOIN ( {self.query_pulse} ) AS query_pulse \
                            NATURAL LEFT JOIN ( {self.grouped_activity}) as activity \
                            where uid = '{user_id}' '''
        elif starttime is None:
            final_query =f'''SELECT * FROM ( {self.query_heart}  )  AS query_heart \
                            NATURAL JOIN ( {self.query_body_battery} ) AS query_body_battery \
                            NATURAL JOIN ( {self.query_pulse} ) AS query_pulse \
                            NATURAL LEFT JOIN ( {self.grouped_activity}) as activity \
                            where uid = '{user_id}' AND time < '{endtime}' '''
        elif endtime is None:
            final_query =f'''SELECT * FROM ( {self.query_heart}  )  AS query_heart \
                            NATURAL JOIN ( {self.query_body_battery} ) AS query_body_battery \
                            NATURAL JOIN ( {self.query_pulse} ) AS query_pulse \
                            NATURAL LEFT JOIN ( {self.grouped_activity}) as activity \
                            where uid = '{user_id}' AND time >= '{starttime}' '''
        else:
            final_query =f'''SELECT * FROM ( {self.query_heart}  )  AS query_heart \
                            NATURAL JOIN ( {self.query_body_battery} ) AS query_body_battery \
                            NATURAL JOIN ( {self.query_pulse} ) AS query_pulse \
                            NATURAL LEFT JOIN ( {self.grouped_activity}) as activity \
                            where uid = '{user_id}' AND time >= '{starttime}' and time < '{endtime}' '''
        cat_score_retrieve_query_prediction = self.cat_score_retrieve_query
        sleep_activity_prediction = self.sleep_activity
        return [final_query, sleep_activity_prediction, cat_score_retrieve_query_prediction]

    def _concat_aggregated_data(self, aggregated_data):
        keys = aggregated_data.keys()
        for hour in keys:
            aggregated_data[hour] = aggregated_data[hour].reset_index(drop=True)
            aggregated_data[hour].columns = [f"{column}_{hour}" for column in aggregated_data[hour].columns]
        return pd.concat(aggregated_data.values(), axis=1)

    def _aggregate(self, data):
        # Quality check - using count
        # Inclusion criterion - CAT > 5 not include that. if CAT score not available, go upto 7 previous day.else discard the data
        # Data Interpolation
        columns = ['heartrate_count', 'heartrate_min', 'heartrate_max', 'heartrate_mean',
                    'heartrate_std', 'heartrate_median', 'heartrate_model', 'heartrate_iqr',
                    'heartrate_skew', 'body_battery_count', 'body_battery_min',
                    'body_battery_max', 'body_battery_mean', 'body_battery_std',
                    'body_battery_median', 'body_battery_model', 'body_battery_iqr',
                    'body_battery_skew', 'pulse_count', 'pulse_min', 'pulse_max',
                    'pulse_mean', 'pulse_std', 'pulse_median', 'pulse_model', 'pulse_iqr',
                    'pulse_skew', 'activity_duration', 'activity_distance', 'activity_calories', 'activity_steps']
        hours = data['hour'].tolist()
        aggregated_data = {}
        for hour in range(24):
            if hour in hours:
                aggregated_data[hour] = data[data['hour'] == hour][columns]
            else:
                aggregated_data[hour] = pd.DataFrame(data=[[0] * len(columns)], columns=columns)
        return self._concat_aggregated_data(aggregated_data)

    def _aggregate_to_daily_data(self, prepared_data):
        # This converts hourly data to  daily data
        # How to handle missing daily data for each variables.
        # Currently replacing all the mising hour data with zero.
        aggregated_data = prepared_data.groupby(["uid", "date"]).apply(self._aggregate)
        return aggregated_data.reset_index()

    def _aggregate_sleep(self, row):
        uid = row["uid"]
        time = row["time"]
        starttime = row['sleep_start_time']
        endtime = row['sleep_end_time']
        sleep_level = row['sleep_level']
        start_hour = starttime.hour
        end_hour = endtime.hour
        if start_hour == end_hour:
            return pd.DataFrame([[uid, starttime.date(), (endtime - starttime).seconds, sleep_level, start_hour]], columns=["uid", "date","duration", "sleep_level", "hour"])
        else:
            current_time = starttime
            sleep_data = []
            sleep_data.append([uid, current_time.date(), 3600 - starttime.minute * 60, sleep_level, start_hour])
            current_hour = start_hour + 1 % 24
            current_time += timedelta(seconds=3600 - starttime.minute * 60)
            while(current_hour < end_hour):
                sleep_data.append([uid, current_time.date(), 3600, sleep_level, current_hour])
                current_hour =  (current_hour + 1) % 24
                current_time += timedelta(seconds=3600 - starttime.minute * 60)
            sleep_data.append([uid, endtime.date(), endtime.minute * 60, sleep_level, end_hour])
            return pd.DataFrame(sleep_data, columns=["uid", "date","duration", "sleep_level", "hour"])

    def _test_apply(self, row):
        sleep_duration_dict =  row[["duration", "sleep_level"]].groupby("sleep_level").sum()["duration"].to_dict()
        sleep_levels = ['light', 'rem', 'awake', 'deep', 'unmeasurable']
        for level in sleep_levels:
            if level not in sleep_duration_dict:
                sleep_duration_dict[level] = 0
        return pd.DataFrame([[sleep_duration_dict[level] for level in sleep_duration_dict]], columns=sleep_levels)

    def _convert_sleep_data_to_hourly(self, sleep_data):
        aggregated_sleep = sleep_data.apply(self._aggregate_sleep, axis=1)
        aggregated_sleep = pd.concat(aggregated_sleep.values, axis=0).reset_index(drop=True)
        print(aggregated_sleep)
        hourly_sleep_data = aggregated_sleep.groupby(["uid", "date", "hour"]).apply(self._test_apply)
        hourly_sleep_data = hourly_sleep_data.reset_index().drop("level_3", axis=1)
        return hourly_sleep_data

    def _create_windowed_data(self, daily_aggregate_data):
        # Take aggregated daily data as input and  return windowed input.
        # TODO: What to do when data for a day is missing? - currently just skipping it
        daily_aggregate_data = daily_aggregate_data.sort_values(["uid", "date"])
        indexer = {}
        index_record = 0
        dataset = []
        for uid in daily_aggregate_data["uid"].unique():
            df = daily_aggregate_data[daily_aggregate_data["uid"] == uid]
            diff_values = df["date"] - df["date"].iloc[0]
            last_value = None
            current_window_size = None
            for idx, value  in enumerate(diff_values):
                if last_value is None or value - last_value != datetime.timedelta(days=1):
                    last_value = value
                    current_window_size = 1
                else:
                    if current_window_size + 1 == self.window_size:
                        dataset.append(df.iloc[idx - self.window_size + 1: idx + 1, 3:].values)
                        indexer[index_record] = (df.iloc[idx, 0], df.iloc[idx, 1])
                        index_record += 1
                        last_value = value
                    else:
                        last_value = value
                        current_window_size += 1
        return np.array(dataset), indexer

    def preprocess_data(self, raw_data):
        hourly_data, sleep_data, cat_score = raw_data
        # Handle missing (NA) data
        if hourly_data.empty:
            return None
        hourly_data = hourly_data.fillna(0)
        hourly_sleep_data = self._convert_sleep_data_to_hourly(sleep_data)
        daily_aggregate_data = self._aggregate_to_daily_data(hourly_data)
        windowed_data, windowed_data_index = self._create_windowed_data(daily_aggregate_data)
        if windowed_data.shape[0] == 0:
            return None
        # Currently dropping window but might be usefull in the future.
        # daily_aggregate_data = daily_aggregate_data.drop(['uid', 'window_end_date', 'level_2'],axis=1 )
        return windowed_data, windowed_data_index