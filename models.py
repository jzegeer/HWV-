#Jake Zegeer
#G#01056701

import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB

class Models:

    def __init__(self):
        self.train_df = None
        self.test_df = None


    def readData(self):
        #read the data from training
        self.train_df = pd.read_csv('data/train.csv',header=None)
        #read the data from testing
        self.test_df = pd.read_csv('data/test_no_label.csv',header=None)

        #dataFrame to list from training withpout labels 
        self.train_x = self.train_df.iloc[:,:-1].values
        #dataFrame to list from training only labels
        self.train_y = self.train_df.iloc[:,-1:].values
        #dataFrame to list from testing
        self.test_x = self.test_df.iloc[:,:].values

    def show_Cross_Validation(self,result):
        for i , j in enumerate(result):
            print(j,' ',result[j])

    def DecisionTree(self):
        #criterion parameter to measure the quality of a split
        #max_depth is use for maximum depth of the tree
        clf_DTC = DecisionTreeClassifier(criterion='entropy',max_depth = 54)
        #Train this classifier with data and labels
        clf_DTC.fit(self.train_x,self.train_y)
        #clf_DTC is object to use to fit the data
        #cv is determines the cross valdation with range of cv
        #cross_validate is determine the fit time, score time and test score
        cv_results_DTC = cross_validate(clf_DTC, self.train_x, self.train_y, cv=5)
        print('Decision Tree')
        #show the result of cross validation
        self.show_Cross_Validation(cv_results_DTC)
        #predict all labels of test and add column in test_df of predict values in dataFrame
        self.test_df['DTC_Labels'] = clf_DTC.predict(self.test_x)
        #It convert to list
        lst = list(self.test_df['DTC_Labels'])
        self.writeDotFile(lst,'DTC.dat')


    def NaiveBayes(self):
        clf_NB = BernoulliNB()
        #Train this classifier with data and labels
        clf_NB.fit(self.train_x,self.train_y)
        #clf_NB is object to use to fit the data
        #cv is determines the cross valdation with range of cv
        #cross_validate is determine the fit time, score time and test score
        cv_results_NB = cross_validate(clf_NB, self.train_x, self.train_y, cv=5)
        print('Naive Bayes')
        #show the result of cross validation
        self.show_Cross_Validation(cv_results_NB)
        #predict all labels of test and add column in test_df of predict values in dataFrame
        self.test_df['NB_Labels'] = clf_NB.predict(self.test_x)
        #It convert to list
        lst = list(self.test_df['NB_Labels'])
        self.writeDotFile(lst,'NB.dat')

    
    def KNeighbors(self):
        from sklearn.neighbors import KNeighborsClassifier
        #n_neighbors is number of neighbours
        clf_KNN = KNeighborsClassifier(n_neighbors=3)
        #Train this classifier with data and labels
        clf_KNN.fit(self.train_x,self.train_y)
        #clf_KNN is object to use to fit the data
        #cv is determines the cross valdation with range of cv
        #cross_validate is determine the fit time, score time and test score
        cv_results_KNN = cross_validate(clf_KNN, self.train_x, self.train_y, cv=5)
        print('K Neighbors')
        #show the result of cross validation
        self.show_Cross_Validation(cv_results_KNN)
        #predict all labels of test and add column in test_df of predict values in dataFrame
        self.test_df['KNN_Labels'] = clf_KNN.predict(self.test_x)
        #It convert to list
        lst = list(self.test_df['KNN_Labels'])
        self.writeDotFile(lst,'KNN.dat')

    def writeDotFile(self,lst,filename):
        #it create the dataframe
        output_df = pd.DataFrame(lst)  
        #it write in .dat format file
        output_df.to_csv(filename,header=False,index=False)

    def writeTestWithLabels(self):
        #write the dataframe into csv file with also add the labels of three classfier
        self.test_df.to_csv('test_with_labels.csv')


if __name__ == "__main__":
    obj = Models()
    obj.readData()
    obj.DecisionTree()
    obj.NaiveBayes()
    obj.KNeighbors()
    obj.writeTestWithLabels()
