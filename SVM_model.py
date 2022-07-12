from sklearn import svm
from loaddata import load_eeg_for_svm

pre_process = './SEED/Preprocessed_EEG/'
extracted = './SEED/ExtractedFeatures/'

data_for_use, label_for_use = load_eeg_for_svm(data_path=r'./SEED/ExtractedFeatures/1_20131027.mat')
tr_data = data_for_use[0]
tr_label = label_for_use[0]
val_data = data_for_use[1]
val_label = label_for_use[1]
te_data = data_for_use[2]
te_label = label_for_use[2]

clf = svm.SVC(C=0.8,kernel='linear',gamma=20,decision_function_shape='ovo')
clf.fit(tr_data,tr_label)

print(clf.score(tr_data,tr_label))
print(clf.score(val_data,val_label))
print(clf.score(te_data,te_label))