from model_class import  ModelClass
import pandas as pd

class LungStudy(ModelClass):

    def __init__(self):
        super().__init__()
        self.window_size = 5
        self.query_heart = '''SELECT "PROJECTID"  as pid, \
            "USERID" as uid, \
            "SOURCEID"  as sid,\
            "TIME"  as time,\
            "WINDOW_START" as window_start,\
            "WINDOW_END" as window_end,\
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
            "TIME"  as time, \
            "WINDOW_START" as window_start, \
            "WINDOW_END" as window_end, \
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
            "TIME"  as time, \
            "WINDOW_START" as window_start, \
            "WINDOW_END" as window_end, \
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

    def get_query_for_training(self):
        final_query =f'''SELECT * FROM ( {self.query_heart}  )  AS query_heart \
                        NATURAL JOIN ( {self.query_body_battery} ) AS query_body_battery \
                        NATURAL JOIN ( {self.query_pulse} ) AS query_pulse'''
        return final_query


    def get_query_for_prediction(self, user_id, project_id, starttime, endtime):

        if starttime is None and endtime is None:
            final_query =f'''SELECT * FROM ( {self.query_heart}  )  AS query_heart \
                            NATURAL JOIN ( {self.query_body_battery} ) AS query_body_battery \
                            NATURAL JOIN ( {self.query_pulse} ) AS query_pulse \
                            where uid = {user_id} AND pid = {project_id}'''
        elif starttime is None:
            final_query =f'''SELECT * FROM ( {self.query_heart}  )  AS query_heart \
                            NATURAL JOIN ( {self.query_body_battery} ) AS query_body_battery \
                            NATURAL JOIN ( {self.query_pulse} ) AS query_pulse \
                            where uid = {user_id} AND pid = {project_id} AND time >= {starttime}'''
        elif endtime is None:
            final_query =f'''SELECT * FROM ( {self.query_heart}  )  AS query_heart \
                            NATURAL JOIN ( {self.query_body_battery} ) AS query_body_battery \
                            NATURAL JOIN ( {self.query_pulse} ) AS query_pulse \
                            where uid = {user_id} AND pid = {project_id} AND time < {endtime}'''
        else:
            final_query =f'''SELECT * FROM ( {self.query_heart}  )  AS query_heart \
                            NATURAL JOIN ( {self.query_body_battery} ) AS query_body_battery \
                            NATURAL JOIN ( {self.query_pulse} ) AS query_pulse \
                            where uid = {user_id} AND pid = {project_id} AND (time between {starttime} and {endtime})'''
        return final_query

    def _concat_aggregated_data(self, aggregated_data):
        keys = aggregated_data.keys()
        for hour in keys:
            aggregated_data[hour] = aggregated_data[hour].reset_index(drop=True)
            aggregated_data[hour].columns = [f"{column}_{hour}" for column in aggregated_data[hour].columns]
        return pd.concat(aggregated_data.values(), axis=1)

    def _aggregate(self, data):
        columns = ['heartrate_count', 'heartrate_min', 'heartrate_max', 'heartrate_mean',
                    'heartrate_std', 'heartrate_median', 'heartrate_model', 'heartrate_iqr',
                    'heartrate_skew', 'body_battery_count', 'body_battery_min',
                    'body_battery_max', 'body_battery_mean', 'body_battery_std',
                    'body_battery_median', 'body_battery_model', 'body_battery_iqr',
                    'body_battery_skew', 'pulse_count', 'pulse_min', 'pulse_max',
                    'pulse_mean', 'pulse_std', 'pulse_median', 'pulse_model', 'pulse_iqr',
                    'pulse_skew']
        hours = data['window_end'].dt.hour.tolist()
        aggregated_data = {}
        for hour in range(24):
            if hour in hours:
                aggregated_data[hour] = data[data['window_end'].dt.hour == hour][columns]
            else:
                aggregated_data[hour] = pd.DataFrame(data=[[0] * len(columns)], columns=columns)
        return self._concat_aggregated_data(aggregated_data)

    def _aggregate_to_daily_data(self, prepared_data):
        # This converts hourly data to  daily data
        # How to handle missing daily data for each variables.
        # Currently replacing all the mising hour data with zero.
        prepared_data["window_end_date"] = prepared_data["window_end"].dt.date
        aggregated_data = prepared_data.groupby(["uid", "window_end_date"]).apply(self._aggregate)
        return aggregated_data.reset_index()

    def _create_windowed_data(self, daily_aggregate_data):
        # Take aggregated daily data as input and  return windowed input.
        daily_aggregate_data.groupby("uid")

    def preprocess_data(self, prepared_data):
        prepared_data["window_start"] = pd.to_datetime(prepared_data['window_start'],unit='ms')
        prepared_data["window_end"] =  pd.to_datetime(prepared_data['window_end'],unit='ms')
        # Handle missing (NA) data
        prepared_data = prepared_data.fillna(0)
        daily_aggregate_data = self._aggregate_to_daily_data(prepared_data)
        #windowed_data = self._create_windowed_data(daily_aggregate_data)
        # Currently dropping window but might be usefull in the future.
        daily_aggregate_data = daily_aggregate_data.drop(['uid', 'window_end_date', 'level_2'],axis=1 )
        return daily_aggregate_data