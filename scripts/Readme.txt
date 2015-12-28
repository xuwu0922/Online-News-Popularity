# Online news popularity
# created by: Xu Wu

Three scprits included for model construction, testing, performance experiment:

Before run the scripts, please install following packages:
	1: pandas
	2. scikit-learn
	3. numpy
	4. xlsxwriter

OnlineNewsPopularity.csv:
	orignial provided dataset

performanceCaculation.py: 
	1.construct models based on 4 algorithms(Decision Tree, KNN, Random Forest, Naive Bayes), collect accuracy and time elapsed for every model.

	2.must run in the same folder with the provided dataset "OnlineNewsPopularity.csv"

changeTrain.py:
	1.keep testing data set unchanged(40% of original dataset), increase the size of training set 5% of original training set, and then record the accuracy and time elapsed every increase round, write result list to excel files

	2. output "changeTrain_output.xlsx" in the same folder

changeTest.py:
	1.keep training data set unchanged(60% of original dataset), increase the size of testing set 5% of original testing set, and then record the accuracy and time elapsed every increase round, write result list to excel files

	2. output "changeTest_output.xlsx" in the same folder

changeTest_output_sample.xlsx
changeTrain_output_sample.xlsx
	1.Two sample outputfiles including data from above two scripts' output, also plot result graph, which you can see in the report

	2.Note: the time elapsed may vary if you try to run scripts in different environments

For more details, please see the report, thank you! 
