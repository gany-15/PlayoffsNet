# import json
from flask import Blueprint, render_template, request, flash, redirect, url_for
from sklearn.model_selection import train_test_split
import pickle
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score
# from sklearn.svm import SVC
# from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# import csv
# import NBABackend
# import requests
import tensorflow as tf
import shap
from shap.plots._force_matplotlib import draw_additive_plot

auth = Blueprint("auth", __name__)

df2 = pd.read_csv("NBABackend/csv/Player Totals.csv")

currentTeamComparison = []

plotColumns = ['fg', 'fga', 'fg_percent', 'x3p', 'x3pa', 'x3p_percent', 'x2p', 'x2pa',
       'x2p_percent', 'e_fg_percent', 'ft', 'fta', 'orb', 'drb', 'trb', 'ast',
       'stl', 'blk', 'tov', 'pf', 'pts', 'teamChemistry']

@auth.route("/",  methods = ["GET", "POST"])
def login():
    # print("Here")
    if request.method == "POST":

        # dummyRow = [0] * 22
        # cols = ['season', 'fg', 'fga', 'fg_percent', 'x3p', 'x3pa', 'x3p_percent', 'x2p', 'x2pa', 'x2p_percent', 'e_fg_percent', 'ft', 'fta', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts', 'teamChemistry']

        def sample_rows(group):
            true_rows = group[group['playoffs'] == True].sample(n=2, random_state=25)
            false_rows = group[group['playoffs'] == False].sample(n=2, random_state=25)
            return pd.concat([true_rows, false_rows])

        df = pd.read_csv('NBABackend/csv/latestOutputWithNew2.csv')
        df = df[df['season_column'] < 2023][df['season_column'] > 1979]
        df.drop(df.columns[0], axis=1, inplace=True)
        train_df = df.drop(['playoffs', 'season_column', 'tm'], axis=1)
        scaler = StandardScaler()
        train_df = scaler.fit_transform(train_df)
        train_df = pd.DataFrame(train_df, index=df.index, columns=df.drop(['playoffs', 'season_column', 'tm'], axis=1).columns)
        df[train_df.columns] = train_df
        df['playoffs'] = df['playoffs'].map({True: float(1), False: float(0)})

        sampled_df = df.groupby('season_column', group_keys=False).apply(sample_rows)
        remaining_df = df[~df.index.isin(sampled_df.index)]
        

        team = request.form.get('team')
        seas = request.form.get('season')
        data_df = df[df['season_column'] == np.int64(seas)][df['tm'] == team]
        print('season', df.iloc[0]['season_column'], type(df.iloc[0]['season_column']), type(np.int64(seas)))

        sampled_df.drop(['tm'], axis=1, inplace=True)
        remaining_df.drop(['tm'], axis=1, inplace=True)
        ip_data = data_df.drop(['season_column', 'tm', 'playoffs'], axis=1)

        # ip_data = pd.DataFrame(scaler.transform(ip_data), columns=ip_data.columns)
        print('\nDATA ROW\n', ip_data)
        model = tf.keras.models.load_model('NBABackend/static/tanh_model_86p.h5')
        result = model.predict(ip_data)
        print('\nRESULT=', result[0])
        returnText = 'False' if result[0] < 0.5 else 'True'

        shp = generatePlot(model, sampled_df, remaining_df, ip_data)
       
        return render_template('result.html', text=returnText, shp=shp)

    return render_template("/project.html", text = "Trial")

def generatePlot(model, remaining_df, sampled_df, ip_data):
    explainer = shap.DeepExplainer(model, remaining_df.drop(['season_column', 'playoffs'], axis=1).values)
    shap_values = explainer.shap_values(sampled_df.drop(['season_column', 'playoffs'], axis=1).values)

    shap.initjs()
    sh = shap.force_plot(explainer.expected_value.numpy(), np.transpose(shap_values[2]), ip_data)
    
    shap_plot = f"<head>{shap.getjs()}</head><body>{sh.html()}</body>"
    return shap_plot


def calculate_win_percentage(wins_losses):
    num_wins = wins_losses.count('W')
    num_losses = wins_losses.count('L')
    win_percentage = num_wins / (num_wins + num_losses)
    return win_percentage


@auth.route("/loadGame", methods = ["GET", "POST"])
def loadGame():
    dataSetGame = pd.read_csv("NBABackend/csv/game.csv")
    data_shape = dataSetGame.shape
    teamNames = ['MIL', 'TOT', 'MIN', 'DAL', 'DEN', 'ATL', 'IND', 'OKC', 'ORL',
       'BOS', 'DET', 'CHI', 'SAC', 'SAS', 'BRK', 'HOU', 'LAC', 'GSW',
       'POR', 'LAL', 'WAS', 'MIA', 'PHO', 'MEM', 'NOP', 'CHO', 'NYK',
       'CLE', 'TOR', 'UTA', 'PHI']
    dataSetGame = dataSetGame[dataSetGame["team_abbreviation_home"].isin(teamNames)]
    dataSetGame = dataSetGame[dataSetGame["game_date"] >="2015-01-01 00:00:00"]
    dataSetGame = dataSetGame.drop(columns = ["season_id", "team_id_home", "game_id", "min", "plus_minus_home", "video_available_home",
                                         "plus_minus_away", "video_available_away"])
    dataSetGame = dataSetGame.drop(columns = ['wl_away'])
    dataSetGame = dataSetGame.dropna()
    matchDataSet = dataSetGame.groupby(['team_abbreviation_home',"team_abbreviation_away"])['wl_home'].sum()    
    newMatchDataSet = pd.DataFrame(matchDataSet) 
    newMatchDataSet['winPercentage'] = newMatchDataSet['wl_home'].apply(calculate_win_percentage)

    teamList = np.array(newMatchDataSet.index[:])
    homelist = [t[0] for t in teamList]
    awaylist = [t[1] for t in teamList]
    newerDataSet = pd.DataFrame({"home": homelist, "away": awaylist, "winPercentage": newMatchDataSet['winPercentage'] })
    # render the template with data
    print(data_shape)
    return render_template('csv.html', text =newerDataSet)
        
@auth.route("/loadplayerTotals", methods = ["GET", "POST"])
def loadplayerTotals():
    dataSet = pd.read_csv("NBABackend/csv/Player Totals.csv")
    dataSet = dataSet[dataSet["lg"] == "NBA"]
    dataSet = dataSet[dataSet["season"] >= 2015]
    dataSet = dataSet.drop(columns = "birth_year")
    teamNames = []
    teamNames = dataSet["tm"].unique()
    missingValues = dataSet.isna().sum().sum()
    columnsMissing = dataSet.columns[dataSet.isna().any()].tolist()
    dataSet = dataSet.drop(columns = "ft_percent")
    dataSet = dataSet.fillna(0)
    columnsMissing = dataSet.columns[dataSet.isna().any()].tolist()
    dataSet = dataSet.drop(columns = ["player_id", "seas_id"])
    playerStastics = dataSet.drop(columns =["g", "gs", 'season', 'player', 'age', 'experience', 'lg', 'tm', "mp"])
    playerStastics["pos"] = playerStastics["pos"].str.split('-').str.get(0)
    middlePlayerStatistics = playerStastics.drop(columns = "pos")
    plotStatics = pd.DataFrame(MinMaxScaler(feature_range=(0, 100)).fit_transform(middlePlayerStatistics), columns=middlePlayerStatistics.columns)
    plotStatics["pos"] = playerStastics["pos"]
    plotStatics = plotStatics.groupby("pos").mean()
    scaledPlotStatistics = plotStatics.drop(columns =["fg_percent", "x3p_percent", "x2p_percent", "e_fg_percent"])
    return render_template('csv.html', text = scaledPlotStatistics)

@auth.route("/loadTeamTotals", methods = ["GET", "POST"])
def loadTeamTotals():
    df = pd.read_csv('NBABackend/csv/Team Totals.csv')
    df = df[df.season >= 2015]
    df = df[df.team != 'League Average']
    playoffs_df = df.groupby('abbreviation')['playoffs'].sum()
    playoffs_df = pd.DataFrame({'Team': playoffs_df.index, 'Count': playoffs_df.values})
    stats_df = df[df.columns[4:]]
    stats_df = stats_df.drop(['g', 'mp'], axis=1)
    stats_df.playoffs = stats_df.playoffs.replace([True, False], [1, 0])
    global df2
    df2 = stats_df
    pl_stats_df = pd.DataFrame(MinMaxScaler(feature_range=(0, 100)).fit_transform(stats_df), columns=stats_df.columns)
    pl_stats_df = pl_stats_df.groupby('playoffs').mean()    
    pl_stats_nop = pl_stats_df.drop(['fg_percent', 'x3p_percent', 'x2p_percent', 'ft_percent'], axis=1)
    return render_template('csv.html', text = stats_df)

@auth.route("/trial", methods = ["GET", "POST"])
def trial():
    global df2
    return render_template('csv.html', text = df2)

@auth.route("/modelTrain", methods = ["GET", "POST"])
def modelTrain():
    # global df2
    # inputDataSet = df2.drop(columns = ["playoffs"])
    # outDataSet = df2["playoffs"]
    # X_train, X_test, Y_train, Y_test = train_test_split( inputDataSet, outDataSet, test_size=0.2)
    # logit=linear_model.LogisticRegression(C=1e10,max_iter=1e5)
    # logit.fit(X_train,Y_train)
    # print('Logistic Regression Accuracy Score on Test data:', logit.score(X_test,Y_test))
    # print('Logistic Regression Accuracy Score on train data:', logit.score(X_train,Y_train))
    
    # predictions = logit.predict(X_test)
    # print(confusion_matrix(Y_test,predictions))
    # print(classification_report(Y_test,predictions))
    # # logit.save("firstTry.h5")
    # with open('my_model.pkl', 'wb') as f:
    #     pickle.dump(logit, f)
    # return render_template('csv.html', text = predictions)
    dataSet = pd.read_csv("NBABackend/csv/Player Totals.csv")
    dataSet = dataSet[dataSet["lg"] == "NBA"]
    dataSet = dataSet[dataSet["season"] >= 2015]
    dataSet = dataSet.drop(columns = "birth_year")
    trialPlayer = dataSet.drop(columns = ["seas_id", "player_id", "age", "experience","lg", "mp","ft_percent", 'g', 'gs'])
    trialPlayer2 = trialPlayer
    stats = {
    'C': [0.7, 0.4, 0.9, 0.3, 0.3, 0.5, 0.8, 0.8, 0.8, 0.6, 0.7, 0.8, 0.8, 0.8, 0.8, 0.4, 0.6, 0.8, 0.8, 0.8, 0.7],
    'PF': [0.6, 0.5, 0.75, 0.5, 0.5, 0.65, 0.6, 0.6, 0.65, 0.5, 0.5, 0.6, 0.6, 0.65, 0.65, 0.4, 0.6, 0.65, 0.75, 0.7, 0.7],
    'PG': [0.8, 0.8, 0.6, 0.65, 0.65, 0.8, 0.6, 0.7, 0.5, 0.5, 0.8, 0.75, 0.4, 0.55, 0.55, 0.8, 0.8, 0.5, 0.7, 0.65, 0.75],
    'SF': [0.4, 0.5, 0.6, 0.65, 0.65, 0.8, 0.4, 0.4, 0.55, 0.5, 0.5, 0.55, 0.45, 0.5, 0.5, 0.4, 0.7, 0.55, 0.7, 0.65, 0.7],
    'SG': [0.5, 0.7, 0.6, 0.75, 0.75, 0.8, 0.4, 0.5, 0.5, 0.5, 0.5, 0.55, 0.4, 0.5, 0.5, 0.5, 0.7, 0.5, 0.8, 0.65, 0.7]
}
    # for index, row in trial.iterrows():
    #     for index2, row2 in chemistryDataFrame.iterrows():
    #         if trial.loc[index, "season"] == chemistryDataFrame.loc[index2, "Year"] and trial.loc[index, "abbreviation"] == chemistryDataFrame.loc[index2, "Team"]:
    #             trial.loc[index, "teamChemistry"] = chemistryDataFrame.loc[index2, "Value"]
    
    trialPlayer2["pos"] = trialPlayer2["pos"].str.split('-').str.get(0)
    trialPlayer2 = trialPlayer2.drop(columns = ['season', 'player', 'pos', 'tm'])
    for index, row in trialPlayer.iterrows():
#     trialPlayer2.loc[index][-21:] = trialPlayer2.loc[index][-21:]*stats[trialPlayer2.loc[index, "pos"]]
        trialPlayer2.loc[index] = trialPlayer2.loc[index] * stats[trialPlayer.loc[index, "pos"]]
    trialPlayer3 = trialPlayer2
    trialPlayer3["season"] = trialPlayer["season"]
    trialPlayer3["player"] = trialPlayer["player"]
    trialPlayer3["pos"] = trialPlayer["pos"]
    trialPlayer3["tm"] = trialPlayer["tm"]
    trialPlayer3 = trialPlayer3.fillna(0)
    trialPlayer4 = trialPlayer3.drop(columns =["pos", "player"])
    team_year_avg = trialPlayer4.groupby(['season', 'tm']).mean()
    df = team_year_avg.reset_index().rename(columns={'season': 'season_column'})
    rows_to_drop = df[df["tm"] == "TOT"].index

# Drop the rows
    df.drop(rows_to_drop, inplace=True)
    rows_to_drop = trialPlayer3[trialPlayer3["tm"] == "TOT"].index

# Drop the rows
    trialPlayer3.drop(rows_to_drop, inplace=True)
    grouped = trialPlayer3.groupby(['season', 'tm'])
    teams_dict = {}

    for (year, team), players in grouped:
        if year not in teams_dict:
            teams_dict[year] = {}
        teams_dict[year][team] = players['player'].tolist()
    
    newtrial = []
    for key in teams_dict:
        for team in teams_dict[key]:
            current = teams_dict[key][team]
            chemistry = 0
            for player in current:
                for year in teams_dict:
                    for key2 in teams_dict[year]:
                        if player in teams_dict[year][key2]:
                            for player2 in current:
                                if player2 in teams_dict[year][key2]:
                                    if player != player2:
                                        chemistry = chemistry + 1
                                    else:
                                        continue
                                else:
                                    continue
            newtrial.append([key, team, chemistry])
    
    columns = ["Year", "Team", "Value"]

    chemistryDataFrame = pd.DataFrame(newtrial, columns=columns)

    chemistryDataFrame[chemistryDataFrame["Team"] == "BOS"]

    for index, row in df.iterrows():
        for index2, row2 in chemistryDataFrame.iterrows():
            if df.loc[index, "season_column"] == chemistryDataFrame.loc[index2, "Year"] and df.loc[index, "tm"] == chemistryDataFrame.loc[index2, "Team"]:
                df.loc[index, "teamChemistry"] = chemistryDataFrame.loc[index2, "Value"]
    
    df = df.fillna(0)
    df2 = pd.read_csv('NBABackend/csv/Team Totals.csv')
    df2 = df2[df2.season >= 2015]
    df2 = df2[df2.team != 'League Average']
    df["playoffs"] = False
    for index, row in df.iterrows():
        for index2, row2 in df2.iterrows():
            if df.loc[index, "season_column"] == df2.loc[index2, "season"] and df.loc[index, "tm"] == df2.loc[index2, "abbreviation"]:
                df.loc[index, "playoffs"] = df2.loc[index2, "playoffs"]

    inputDataSet = df.drop(columns = ["playoffs", "tm", "season_column"])
    outDataSet = df["playoffs"]
    X_train, X_test, Y_train, Y_test = train_test_split( inputDataSet, outDataSet, test_size=0.2)
    logit=linear_model.LogisticRegression(C=1e10,max_iter=1e5)
    logit.fit(X_train,Y_train)
    print('Logistic Regression Accuracy Score on Test data:', logit.score(X_test,Y_test))
    print('Logistic Regression Accuracy Score on train data:', logit.score(X_train,Y_train))

    with open('final4.pkl', 'wb') as f:
        pickle.dump(logit, f)
    return render_template('csv.html', text = logit.score(X_test,Y_test))



@auth.route("/modelPredict", methods = ["GET", "POST"])
def modelPredict():
    global df2
    df2 = df2.drop(columns = ["playoffs"])
    rowTry = df2.iloc[0]
    rowTry = [rowTry]

    print(rowTry)

    with open('my_model.pkl', 'rb') as file:
        model = pickle.load(file)
    prediction = model.predict(rowTry)

    print(prediction)
    return render_template('csv.html', text = prediction)

@auth.route("/teamCompare", methods = ["GET", "POST"])
def teamCompare():
    if request.method == "POST":
        teamSelected = request.form.get("team")
        global currentTeamComparison

        scale = [6] * 20
        scale += [6, 4]

        df = pd.read_csv('NBABackend/csv/team_compare.csv')
        team1 = scale * df[df['tm'] == teamSelected].values[:, 1:]
        print(team1[0])
        print(scale * currentTeamComparison)
        team1perf = np.sum(team1[0])
        team2 = scale * currentTeamComparison
        team2perf = np.sum(team2)
        winnerTeam = ""
        if team1perf > team2perf: 
            winnerTeam = 'False'
        else:
            winnerTeam = 'True'

        filtcols = df.columns[1:]
        print(filtcols)
        print(team1perf, team2perf)
        generatePlot(team1[0], filtcols, 1)
        generatePlot(team2, filtcols, 2)
        return render_template('teamComparisonResult.html', text = winnerTeam)
    return render_template('teamComparison.html')