import sqlalchemy
import pandas as pd
import numpy as np
import seaborn as sns

from sql_settings import USER, PWD, HOST
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from operator import itemgetter


class FeatureAnalysis(object):
    def __init__(self):
        connection_string = "mysql+pymysql://" + USER + ":" + PWD + "@" + HOST
        self.mysql_engine = sqlalchemy.create_engine(connection_string, echo=False)
        self.users_data = self._read_data()
        self.rf_clf = RandomForestClassifier(n_estimators=200, max_depth=4, oob_score=True, class_weight="auto")

    def _read_data(self):
        """
        Read data from a MySQL data base into data frames and determine adopted users
        :return: data frame; user attributes with adoption column
        """
        print "Reading data from MySQL into data frames..."
        df_users = pd.read_sql_table("takehome_users", self.mysql_engine)
        df_engagement = pd.read_sql_table("takehome_user_engagement", self.mysql_engine)
        print "Determining adopted users..."
        adopted_users = self.determine_adopted_users(df_engagement, num_of_visits=3)
        adopted_users["adopted"] = 1
        df_users_adopted = pd.merge(df_users, adopted_users, left_on="object_id", right_on="adopted_user_id", how="left")
        df_users_adopted.adopted.fillna(0, inplace=True)
        print "Done reading in data."
        return df_users_adopted

    def determine_adopted_users(self, df, num_of_visits):
        """
        Determine whether or not a user can be considered adopted.
        :param num_of_visits: int; number of visits within 7 days for a user to be considered adopted
        :param df:data frame; user statistics
        :return:
        """
        df["time_stamp"] = pd.to_datetime(df["time_stamp"])
        df_grouped = df.groupby('user_id').apply(self._grouper)
        temp = df_grouped[df_grouped["cum_visits"] >= 3]
        adopted = [int(x) for x in np.unique(temp["user_id"].values)]
        adopted = pd.DataFrame(adopted, columns=["adopted_user_id"])
        return adopted

    @staticmethod
    def _grouper(df):
        """
        Helper function to calculate cumulative visits in during a given time frame
        :param df: pandas data frame; user engagement data
        :return: data frame with cumulative visits during a rolling window of length "window"
        """
        df = df.set_index('time_stamp').resample('D', 'last')
        df['cum_visits'] = pd.rolling_sum(df['visited'], 7, 0)
        return df[df.visited.notnull()]


    def preprocess_features(self):
        """
        Preprocess user data, create dummy variables for categorical features. Drop columns like name/email/etc
        :return: --
        """
        print "Preprocessing features..."
        # Encode the categorical features
        dummies = pd.get_dummies(self.users_data["creation_source"])
        self.users_data = pd.concat([self.users_data, dummies], axis=1)

        # Encode the invited_by_user_id column to [0,1] --> not invited/invited respectively
        self.users_data.loc[:, "invited_by_user_id"].fillna(0, inplace=True)
        self.users_data.loc[self.users_data["invited_by_user_id"] != 0, "invited_by_user_id"] = 1

        # Drop features unlikely to have much signal, like names, email addresses, etc
        # However some of these may be useful in feature engineering
        features_to_drop = ["name", "email", "object_id", "creation_time", "adopted_user_id",
                            "creation_source","last_session_creation_time", "org_id"]
        self.users_data.drop(features_to_drop, axis=1, inplace=True)
        print "Done preprocessing.\n"

    def plot_histogram(self):
        """
        Plot histogram of the distribution of features among adopted/not adopted users
        :return: --
        """
        # Split (not) adopted users into separate data frames
        adopted = self.users_data[self.users_data["adopted"] == 1]
        not_adopted = self.users_data[self.users_data["adopted"] == 0]
        # Calculate the percentage of features
        adopted = pd.DataFrame(adopted.apply(lambda x: x.sum()/float(adopted.shape[0]), axis=0))
        not_adopted = pd.DataFrame(not_adopted.apply(lambda x: x.sum()/float(not_adopted.shape[0]), axis=0))
        # Do the plotting
        df_to_plot = pd.concat([adopted,not_adopted], axis=1)
        df_to_plot.columns=["adopted", "not_adopted"]
        df_to_plot.drop("adopted", axis=0, inplace=True)
        df_to_plot.plot(kind="bar")


    def do_random_forest(self):
        """
        Train the random forest classifier and output model performance statistics
        :return: stdout; model performance stats
        """
        # Separate the user data into features and target variables
        y = self.users_data.pop("adopted").values
        X = self.users_data.values
        features = self.users_data.columns

        # Do a stratified split (because the classes are imbalanced, it is preferable to do a stratified train/test split
        split = StratifiedShuffleSplit(y, n_iter=1, test_size=0.20, random_state=123)
        for train_index, test_index in split:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        # Train the model and predict the test set
        self.rf_clf.fit(X_train, y_train)
        preds = self.rf_clf.predict(X_test)

        # Print model performance stats
        print "Model Out Of Bag Error: {0}\n".format(self.rf_clf.oob_score_)

        print "Precision: {}".format(precision_score(y_test, preds))
        print "Recall: {}".format(recall_score(y_test,preds))
        print "F1_score: {}".format(f1_score(y_test, preds))

        # Get the feature importances and print to stdout
        coeffs = self.rf_clf.feature_importances_
        self._print_results(features, coeffs)

    @staticmethod
    def _print_results(feature_names, coefficients):
        """
        Helper function to print the feature importances to stdout
        :param feature_names: list of feature names
        :param coefficients: feature importances resulting from random forest training
        :return: stdout
        """
        print "\nFeature Importances:"
        results = zip(list(feature_names), list(coefficients))
        for tup in sorted(results, key=itemgetter(1), reverse=True):
            print "{0}; {1}".format(tup[0], tup[1])

if __name__ == "__main__":
    fa = FeatureAnalysis()
    fa.preprocess_features()
    fa.plot_histogram()
    fa.do_random_forest()
