from model_class import  ModelClass
import pandas as pd
from datetime import  timedelta
import datetime
from datetime import datetime as dt
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

        self.query_respiration =  '''SELECT "PROJECTID"  as pid, \
            "USERID" as uid, \
            "SOURCEID"  as sid, \
            to_timestamp("TIME"  / 1000 )  as time, \
            extract(hour from to_timestamp("TIME" / 1000)) as hour,\
            date(to_timestamp("TIME"  / 1000 ) ) as date,\
            to_timestamp("WINDOW_START" / 1000 ) as window_start,\
            to_timestamp("WINDOW_END" / 1000 )  as window_end,
            "COUNT" as respiration_count, \
            "MIN" as respiration_min, \
            "MAX" as respiration_max, \
            "MEAN" as respiration_mean, \
            "STDDEV" as respiration_std, \
            "MEDIAN" as respiration_median, \
            "MODE"  as respiration_model, \
            "IQR" as respiration_iqr, \
            "SKEW" as respiration_skew \
            from "PUSH_GARMIN_RESPIRATION_TIMESTAMP_LONG_WINDOWED_1H_TABLE"'''

        self.query_stress = '''SELECT "PROJECTID"  as pid, \
            "USERID" as uid, \
            "SOURCEID"  as sid,\
            to_timestamp("TIME"  / 1000 )   as time,\
            extract(hour from to_timestamp("TIME" / 1000)) as hour,\
            date(to_timestamp("TIME"  / 1000 ) ) as date,\
            to_timestamp("WINDOW_START" / 1000 ) as window_start,\
            to_timestamp("WINDOW_END" / 1000 )  as window_end,\
            "COUNT" as stress_count,\
            "MIN" as stress_min,\
            "MAX" as stress_max,\
            "MEAN" as stress_mean,\
            "STDDEV" as stress_std,\
            "MEDIAN" as stress_median,\
            "MODE"  as stress_model,\
            "IQR" as stress_iqr,\
            "SKEW" as stress_skew \
            from "PUSH_GARMIN_HEART_RATE_SAMPLE_TIMESTAMP_LONG_WINDOWED_1H_TABLE"'''

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

        self.daily_activity_query = f'''SELECT "userId" as uid, \
            "projectId" as pid, \
            "time" as time,\
            "date" as date,\
            "duration", \
            "steps", \
            "distance", \
            "moderateIntensityDuration", \
            "vigorousIntensityDuration", \
            "minHeartRate", \
            "averageHeartRate", \
            "maxHeartRate", \
            "restingHeartRate", \
            "averageStressLevel", \
            "maxStressLevel", \
            "stressDuration", \
            "restStressDuration", \
            "activityStressDuration", \
            "lowStressDuration", \
            "mediumStressDuration", \
            "highStressDuration", \
            "stressQualifier" \
            from "push_garmin_daily_summary" as daily_activity_summary'''

        self.inference_table_name = "inference"
        self.project_id = "RALPMH-COPD-Lon-s1"

    def get_query_for_training(self):
        final_query =f'''SELECT * FROM ( {self.query_heart}  )  AS query_heart \
                        NATURAL JOIN ( {self.query_body_battery} ) AS query_body_battery \
                        NATURAL LEFT JOIN ( {self.query_pulse} ) AS query_pulse \
                        NATURAL JOIN ( {self.query_respiration} ) AS query_respiration \
                        NATURAL LEFT JOIN ( {self.grouped_activity}) as activity \
                        NATURAL LEFT JOIN ( {self.query_stress}) as query_stress \
                        where pid = '{self.project_id}' '''
        return [final_query, self.sleep_activity, self.cat_score_retrieve_query, self.daily_activity_query]


    def get_query_for_prediction(self, user_id, project_id, starttime, endtime):
        if starttime is None and endtime is None:
            final_query =f'''SELECT * FROM ( {self.query_heart}  )  AS query_heart \
                            NATURAL JOIN ( {self.query_body_battery} ) AS query_body_battery \
                            NATURAL LEFT JOIN ( {self.query_pulse} ) AS query_pulse \
                            NATURAL JOIN ( {self.query_respiration} ) AS query_respiration \
                            NATURAL LEFT JOIN ( {self.grouped_activity}) as activity \
                            NATURAL LEFT JOIN ( {self.query_stress}) as query_stress \
                            where uid = '{user_id}' AND pid = '{project_id}' '''
            sleep_activity_query = f'''SELECT * from ({self.sleep_activity}) as sleep_activity where uid = '{user_id}' '''
            daily_activity_query = f'''SELECT * from ({self.daily_activity_query}) as daily_activity_summary where uid = '{user_id}' '''
        elif starttime is None:
            final_query =f'''SELECT * FROM ( {self.query_heart}  )  AS query_heart \
                            NATURAL JOIN ( {self.query_body_battery} ) AS query_body_battery \
                            NATURAL LEFT JOIN ( {self.query_pulse} ) AS query_pulse \
                            NATURAL JOIN ( {self.query_respiration} ) AS query_respiration \
                            NATURAL LEFT JOIN ( {self.grouped_activity}) as activity \
                            NATURAL LEFT JOIN ( {self.query_stress}) as query_stress \
                            where uid = '{user_id}' AND pid = '{project_id}' AND time < '{endtime}' '''
            sleep_activity_query = f'''SELECT * from ({self.sleep_activity}) as sleep_activity where uid = '{user_id}' AND time < '{endtime}' '''
            daily_activity_query = f'''SELECT * from ({self.daily_activity_query}) as daily_activity_summary where uid = '{user_id}' AND time < '{endtime}' '''
        elif endtime is None:
            starttime_with_lag = starttime - timedelta(days=self.window_size - 1)
            final_query =f'''SELECT * FROM ( {self.query_heart}  )  AS query_heart \
                            NATURAL JOIN ( {self.query_body_battery} ) AS query_body_battery \
                            NATURAL LEFT JOIN ( {self.query_pulse} ) AS query_pulse \
                            NATURAL JOIN ( {self.query_respiration} ) AS query_respiration \
                            NATURAL LEFT JOIN ( {self.grouped_activity}) as activity \
                            NATURAL LEFT JOIN ( {self.query_stress}) as query_stress \
                            where uid = '{user_id}' AND pid = '{project_id}' AND time >= '{starttime_with_lag}' '''
            sleep_activity_query = f'''SELECT * from ({self.sleep_activity}) as sleep_activity where uid = '{user_id}' AND time >= '{starttime_with_lag}' '''
            daily_activity_query = f'''SELECT * from ({self.daily_activity_query}) as daily_activity_summary where uid = '{user_id}' AND time >= '{starttime_with_lag}' '''
        else:
            starttime_with_lag = starttime - timedelta(days=self.window_size - 1)
            final_query =f'''SELECT * FROM ( {self.query_heart}  )  AS query_heart \
                            NATURAL JOIN ( {self.query_body_battery} ) AS query_body_battery \
                            NATURAL LEFT JOIN ( {self.query_pulse} ) AS query_pulse \
                            NATURAL JOIN ( {self.query_respiration} ) AS query_respiration \
                            NATURAL LEFT JOIN ( {self.grouped_activity}) as activity \
                            NATURAL LEFT JOIN ( {self.query_stress}) as query_stress \
                            where uid = '{user_id}' AND pid = '{project_id}' AND time >= '{starttime_with_lag}' and time < '{endtime}' '''

            sleep_activity_query = f'''SELECT * from ({self.sleep_activity}) as sleep_activity where uid = '{user_id}' AND time >= '{starttime_with_lag}' and time < '{endtime}' '''
            daily_activity_query = f'''SELECT * from ({self.daily_activity_query}) as daily_activity_summary where uid = '{user_id}' AND time >= '{starttime_with_lag}' and time < '{endtime}' '''
        cat_score_retrieve_query = f'''SELECT * from ({self.cat_score_retrieve_query}) as cat_score where uid = '{user_id}' '''
        return [final_query, sleep_activity_query, cat_score_retrieve_query, daily_activity_query]

    def _concat_aggregated_data(self, aggregated_data):
        keys = aggregated_data.keys()
        for hour in keys:
            aggregated_data[hour] = aggregated_data[hour].reset_index(drop=True)
            aggregated_data[hour].columns = [f"{column}_{hour}" for column in aggregated_data[hour].columns]
        return pd.concat(aggregated_data.values(), axis=1)

    def _aggregate(self, data):
        # Quality check - using count
        # Data Interpolation
        columns = ['heartrate_count', 'heartrate_min', 'heartrate_max', 'heartrate_mean',
                    'heartrate_std', 'heartrate_median', 'heartrate_model', 'heartrate_iqr',
                    'heartrate_skew', 'body_battery_count', 'body_battery_min',
                    'body_battery_max', 'body_battery_mean', 'body_battery_std',
                    'body_battery_median', 'body_battery_model', 'body_battery_iqr',
                    'body_battery_skew', 'pulse_count', 'pulse_min', 'pulse_max',
                    'pulse_mean', 'pulse_std', 'pulse_median', 'pulse_model', 'pulse_iqr',
                    'pulse_skew','respiration_count', 'respiration_min', 'respiration_max',
                    'respiration_mean', 'respiration_std', 'respiration_median', 'respiration_model',
                    'respiration_iqr', 'respiration_skew', "stress_count", 'stress_min', 'stress_max',
                    'stress_mean', 'stress_std', 'stress_median', 'stress_model', 'stress_iqr',
                    'stress_skew', 'activity_duration', 'activity_distance', 'activity_calories', 'activity_steps',
                    'light', 'rem', 'awake', 'deep', 'unmeasurable']
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

        # Inclusion criterion - CAT > 5 not include that. if CAT score not available, go upto 7 previous day.else discard the data
        prepared_data = prepared_data[prepared_data["cat_score"] <= 20]
        aggregated_data = prepared_data.groupby(["uid", "date", "pid"]).apply(self._aggregate)
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
        hourly_sleep_data = aggregated_sleep.groupby(["uid", "date", "hour"]).apply(self._test_apply)
        hourly_sleep_data = hourly_sleep_data.reset_index().drop("level_3", axis=1)
        return hourly_sleep_data

    def _create_windowed_data(self, daily_aggregate_data):
        # Take aggregated daily data as input and  return windowed input.
        # TODO: What to do when data for a day is missing? - currently just skipping it
        daily_aggregate_data = daily_aggregate_data.fillna(-1)
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
                        dataset.append(df.iloc[idx - self.window_size + 1: idx + 1, 4:].values)
                        indexer[index_record] = (df.iloc[idx, 0], df.iloc[idx, 1], df.iloc[idx, 2])
                        index_record += 1
                        last_value = value
                    else:
                        last_value = value
                        current_window_size += 1
        return np.array(dataset), indexer

    def _check_completion(self, hourly_data):
        # Including the completeness of the HR, pulse OX and body battery data as inclusion
        ## Only accept reasonable count of values per hour - 25% in all cases, pulse - 1, respiration - 1
        hourly_data = hourly_data[(hourly_data["heartrate_count"] >= 60) & (hourly_data["body_battery_count"] >= 18)
                    & (hourly_data["respiration_count"] >= 1)].reset_index(drop=True)
        # for HR at least 8 hours in a day, pulse Ox at least 6 hours and body battery at least 8 hours, respiration = 6
        prepared_data_hourly_count = hourly_data.groupby(['uid', "date"]).agg({"heartrate_count": "count", "pulse_count": "count", "body_battery_count": "count", "respiration_count": "count"}).reset_index()
        acceptible_values = prepared_data_hourly_count[(prepared_data_hourly_count["heartrate_count"] >= 8) & (prepared_data_hourly_count["body_battery_count"] >= 8)  & (prepared_data_hourly_count["respiration_count"] >= 6)]
        hourly_data = hourly_data[(hourly_data["uid"].isin(acceptible_values["uid"])) & (hourly_data["date"].isin(acceptible_values["date"]))].reset_index(drop=True)
        return hourly_data

    def _convert_stress_qualifier_to_classes(self, stress_qualifier:pd.Series):
        classes = ["unknown", "calm", "balanced", "stressful", "very_stressful", "calm_awake", "balanced_awake", "stressful_awake", "very_stressful_awake"]
        one_hot = pd.get_dummies(stress_qualifier.astype(pd.CategoricalDtype(categories=classes)), columns=classes, prefix="stressQualifier")
        one_hot = one_hot.drop("stressQualifier_unknown", axis=1)
        return one_hot

    def _preprocess_daily_activity_summary(self, daily_activity_summary):
        daily_activity_summary['date'] = pd.to_datetime(daily_activity_summary['date'])
        one_hot = self._convert_stress_qualifier_to_classes(daily_activity_summary['stressQualifier'])
        daily_activity_summary = daily_activity_summary.join(one_hot).drop(["stressQualifier", "time"], axis=1)
        daily_activity_summary = daily_activity_summary.fillna(-1)
        return daily_activity_summary

    def preprocess_data(self, raw_data, is_inference=False):
        hourly_data, sleep_data, cat_score, daily_activity_summary = raw_data
        # Handle missing (NA) data
        if hourly_data.empty:
            return None
        hourly_data = self._check_completion(hourly_data)
        hourly_data = hourly_data.fillna(-1)
        # Merging CAT data with hourly data.
        hourly_data = hourly_data.merge(cat_score[["uid", "date", "cat_score"]], on=["uid", "date"], how="left")
        hourly_data = hourly_data.sort_values(by=["uid", "date"]).reset_index(drop=True)
        hourly_data["cat_score"] = hourly_data.groupby("uid")["cat_score"].ffill()
        hourly_data.dropna(subset=["cat_score"]).reset_index(drop=True)
        # Converting sleep data to hourly sleep data
        hourly_sleep_data = self._convert_sleep_data_to_hourly(sleep_data)
        hourly_data["hour"] = hourly_data["hour"].astype(int)

        # Merging hourly sleep data with hourly data
        hourly_data = hourly_data.merge(hourly_sleep_data, on=["uid", "hour", "date"], how="left").fillna(-1)
        daily_aggregate_data = self._aggregate_to_daily_data(hourly_data)
        daily_aggregate_data['date'] = pd.to_datetime(daily_aggregate_data['date'])
        daily_activity_summary = self._preprocess_daily_activity_summary(daily_activity_summary)
        # Merging daily activity summary with daily aggregate data on columns "uid", pid and "date"
        daily_aggregate_data = daily_aggregate_data.merge(daily_activity_summary, on=["uid", "pid", "date"], how="left")
        if daily_aggregate_data.empty:
            return None
        windowed_data, windowed_data_index = self._create_windowed_data(daily_aggregate_data)
        if windowed_data.shape[0] <= 10 and not is_inference:
            return None
        # Currently dropping window but might be usefull in the future.
        # daily_aggregate_data = daily_aggregate_data.drop(['uid', 'window_end_date', 'level_2'],axis=1 )
        return windowed_data, windowed_data_index

    def create_return_obj(self, indexes, model_name, model_version, alias, inference_results):
        dateTimeObj = dt.now(tz=None)
        return_obj = pd.DataFrame.from_dict(indexes, orient="index", columns=["uid", "date", "pid"])
        # check if infererence_results is  a tuple
        if isinstance(inference_results, tuple):
            return_obj["invocation_result"] = [{"anomaly_detected": result[0], "output_vector": result[1]} for result in zip(*inference_results)]
        else:
            return_obj["invocation_result"] = [{"anomaly_detected": result} for result in inference_results]

        return_obj["model_name"] = model_name
        return_obj["model_version"] = model_version
        return_obj["alias"] = alias
        return_obj["timestamp"] = dateTimeObj.timestamp()
        return_obj["window_size"] = self.window_size
        return_obj['invocation_result'] = return_obj.invocation_result.map(self.dict2json)
        return return_obj