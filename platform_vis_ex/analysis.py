import json
import altair as alt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def createChart(data, name):
    color_expression    = "highlight._vgsid_==datum._vgsid_"
    color_condition     = alt.ConditionalPredicateValueDef(color_expression, "SteelBlue")
    highlight_selection = alt.selection_single(name="highlight", empty="all", on="mouseover")
    rating_selection    = alt.selection_single(name="rating", empty="all", encodings=['y'])
    maxCount            = int(data['rating'].value_counts().max())

    barMean = alt.Chart() \
        .mark_bar(stroke="Black") \
        .encode(
            alt.X("mean(rating):Q", axis=alt.Axis(title="Rating")),
            alt.Y('name:O', axis=alt.Axis(title="{} App Name".format(name)),
                  sort=alt.SortField(field="rating", op="mean", order='descending')),
            alt.ColorValue("LightGrey", condition=color_condition),
        ).properties(
            selection = highlight_selection+rating_selection,
        )

    barRating = alt.Chart() \
        .mark_bar(stroke="Black") \
        .encode(
            alt.X("rating:O", axis=alt.Axis(title="Rating"),
                  scale=alt.Scale(type='band', domain=list(range(1,6))),
                 ),
            alt.Y("count()", axis=alt.Axis(title="Number of Ratings"),
                  scale=alt.Scale(domain=(0,maxCount)),
                 ),
        ).properties(
            selection = highlight_selection,
        ).transform_filter(
            rating_selection.ref()
        )

    return alt.hconcat(barMean, barRating,
        data=data,
        title="{} App Ratings".format(name)
    )

def showTopWords(contents):
    from sklearn.feature_extraction.text import TfidfVectorizer
    def removeStopWords(contents):
        import re
        separator = re.compile('\W+')
        stopwords = set(map(lambda x: x.strip(), open('bswords.txt', 'r').readlines()))
        stopwords.add('app')
        words = []
        for line in contents:
            words.append(' '.join(w for w in separator.split(line.lower())
                                  if w and w not in stopwords))
        return words

    words = removeStopWords(contents.dropna())
    tfidf = TfidfVectorizer()
    model = tfidf.fit_transform(words)
    features = tfidf.get_feature_names()
    topWords = {}
    for row in model:
        words = list(map(lambda x: features[x[0]],
                         sorted(filter(lambda x: x[1]>0.2,
                                       enumerate(row.toarray().tolist()[0])),
                                key=lambda x: -x[1])))
        for w in words:
            topWords[w] = topWords.get(w, 0)+1
    dfWords = pd.DataFrame(list(filter(lambda x: x[1]>2, topWords.items())),
                           columns=['word', 'freq'])

    def plotWords(dfWords):
        color_expression    = "highlight._vgsid_==datum._vgsid_"
        color_condition     = alt.ConditionalPredicateValueDef(color_expression, "SteelBlue")
        highlight_selection = alt.selection_single(name="highlight", empty="all", on="mouseover")

        return alt.Chart(dfWords) \
            .mark_bar(stroke="Black") \
            .encode(
                alt.X("freq", axis=alt.Axis(title="Count")),
                alt.Y('word:O', axis=alt.Axis(title="Keyword"),
                      sort=alt.SortField(field="freq", op="max", order='descending')),
                alt.ColorValue("LightGrey", condition=color_condition),
            ).properties(
                selection = highlight_selection
            )

    return plotWords(dfWords)

def loadData():
    # load data was in a different directory
    # so we're using os as a different way of accessing the data

    import os #
    cur_dir = os.path.dirname(__file__)

    ios_reviews = json.load(open(os.path.join(cur_dir,'ios_reviews.json'), 'r'))
    android_reviews = json.load(open(os.path.join(cur_dir, 'android_reviews.json'), 'r'))

    dfIOS = pd.DataFrame(((app, review['rating'], review['review'])
                          for app,reviews in ios_reviews.items()
                          for review in reviews), columns=['name', 'rating', 'content'])
    dfAndroid = pd.DataFrame(((app, review['rating'], review['review'])
                              for app,reviews in android_reviews.items()
                              for review in reviews), columns=['name', 'rating', 'content'])
    return {'ios': dfIOS, 'android': dfAndroid}
