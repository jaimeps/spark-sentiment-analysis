
from pyspark import SparkContext
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF, IDF
import re
from nltk import PorterStemmer

### ADD YOUR KEY HERE ###
AWS_ACCESS_KEY_ID = "XX"
AWS_SECRET_ACCESS_KEY = "XX/XX"


# PARSELINE HELPER FUNCTION ===================================================
def parseline(line):
    parts = line.split(' ')
    parts = [x.lower() for x in parts]
    parts = [re.sub('[^A-Za-z]+', '', x) for x in parts]
    parts = [PorterStemmer().stem_word(x) for x in parts]
    return [x for x in parts if len(x) > 1 if x != 'br']

# TRAINING SET ================================================================

sc = SparkContext()
sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", AWS_ACCESS_KEY_ID)
sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", 
	AWS_SECRET_ACCESS_KEY)

text_negative = sc.textFile("s3n://sent/train_neg.txt")
text_positive = sc.textFile("s3n://sent/train_pos.txt")

train_text = text_negative.union(text_positive)
train_labels = text_negative.map(lambda x: 0.0).union(
	text_positive.map(lambda x: 1.0))

tf = HashingTF().transform(train_text.map(parseline, 
	preservesPartitioning=True))
idf = IDF().fit(tf)
train_tfidf = idf.transform(tf)

training = train_labels.zip(train_tfidf).map(lambda x: LabeledPoint(x[0], 
	x[1]))

model = NaiveBayes.train(training)

# TESTING SET =================================================================

text_negative = sc.textFile("s3n://sent/test_neg.txt")
text_positive = sc.textFile("s3n://sent/test_pos.txt")

test_text = text_negative.union(text_positive)
test_tlabels = text_negative.map(lambda x: 0.0).union(
	text_positive.map(lambda x: 1.0))

tf_test = HashingTF().transform(
    test_text.map(parseline, preservesPartitioning=True))

tfidf_test = idf.transform(tf_test)

labeled_prediction = test_tlabels.zip(model.predict(tfidf_test)).map(
    lambda x: {"actual": x[0], "predicted": x[1]})

accuracy = 1.0 * labeled_prediction.filter(lambda doc: doc["actual"] == 
	doc['predicted']).count() / labeled_prediction.count()

print '\n\n===== ACCURACY', accuracy , '=====\n\n'

# =============================================================================