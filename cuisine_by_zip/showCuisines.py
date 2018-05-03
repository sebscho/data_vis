import altair as alt
import pandas as pd
import requests
import json

def loadData():

    response = requests.get('https://raw.githubusercontent.com/hvo/datasets/master/nyc_restaurants_by_cuisine.json')
    response = response.json()

    dfCuisines = pd.DataFrame(columns =['cuisine', 'perZip', 'total', 'zipCode'])

    for i in range(len(response)):
        df_test = pd.read_json(json.dumps(response[i]))
        df_test['zipCode'] = df_test.index
        dfCuisines = dfCuisines.append(df_test)

    return dfCuisines

def showCuisines(data, zipCode):

    data = data[data['zipCode'] == zipCode]

    barRest = alt.Chart(data) \
        .mark_bar(stroke="Black")\
        .encode(alt.X("perZip:Q", axis=alt.Axis(title="Restaurant Count")),\
                alt.Y("cuisine:O", axis=alt.Axis(title="Cuisine"),\
                sort=alt.SortField(field="perZip", op="sum", order='descending')),\
                alt.ColorValue("LightGrey"))

    return barRest
