# Created by mqgao at 2018/9/7

"""
Feature: #Enter feature name here
# Enter feature description here

Scenario: #Enter scenario name here
# Enter steps here

Test File Location: # Enter

"""
columns = """
content 
location_traffic_convenience
location_distance_from_business_district
location_easy_to_find	
service_wait_time
service_waiters_attitude
service_parking_convenience	
service_serving_speed
price_level
price_cost_effective
price_discount
environment_decoration
environment_noise
environment_space
environment_cleaness
dish_portion
dish_taste
dish_look
dish_recommendation	
others_overall_experience	
others_willing_to_consume_again""".split()

columns = [c.strip() for c in columns]

assert len(columns) == 21


def format_csv_result(csv_result):
    csv_result = csv_result.reset_index()
    csv_result = csv_result[columns]
    return csv_result


if __name__ == '__main__':
    import pandas as pd
    file = '../result/result_1536312122.286963.csv'
    csv = format_csv_result(pd.read_csv(file))
    csv.to_csv(file, index=True)





