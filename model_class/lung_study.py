from model_class import  ModelClass
import pandas as pd

class LungStudy(ModelClass):

    def __init__(self):
        super().__init__()
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

    def preprocess_data(self, data):
        prepared_data = data.drop(["pid", "sid", "uid", 'time'],axis=1 )
        prepared_data["window_start"] = pd.to_datetime(prepared_data['window_start'],unit='ms')
        prepared_data["window_end"] =  pd.to_datetime(prepared_data['window_end'],unit='ms')
        # Currently dropping window but might be usefull in the future.
        prepared_data = prepared_data.drop(["window_start", "window_end"],axis=1 )
        # Handle missing (NA) data
        prepared_data = prepared_data.fillna(0)
        return prepared_data