import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier


# Load training data
scams = pd.read_csv('train_data.txt', sep='\t')
smiles = scams.iloc[:,1]
y = scams.iloc[:,2]


# Calculate RDKit descriptors
descr = Descriptors._descList[0:2] + Descriptors._descList[3:]
calc = [x[1] for x in descr]

def describe(mols):
	descrs = []
	for mol in mols:
		fp = AllChem.GetMorganFingerprintAsBitVect(mol,3,nBits=2048)		
		fp_list = []
		fp_list.extend(fp.ToBitString())
		fp_expl = [float(x) for x in fp_list]
		ds_n = []
		for d in calc:
			v = d(mol)
			if v > np.finfo(np.float32).max:
				ds_n.append(np.finfo(np.float32).max)
			else:
				ds_n.append(np.float32(v))
		
		descrs += [fp_expl + list(ds_n)];
	
	return descrs


mols = [Chem.MolFromSmiles(s) for s in smiles]
fps = np.array(describe(mols))


# Transform training data
def classano(x):
	if x == "AGG":
		return "1"
	elif x == "NONAGG":
		return "0"
	else:
		return "-1"

annoclass = np.array([classano(x) for x in y])
y = annoclass[annoclass != "-1"]
x = fps[annoclass != "-1"]


# Scale data
scaler = MinMaxScaler()
scaler2 = MinMaxScaler().fit(x)
x = scaler2.transform(x)


# DeepSCAMs hyperparameters 
seed = 1234
kf = StratifiedKFold(10, shuffle=True, random_state=seed)
MLP = MLPClassifier(activation='tanh', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(100, 1000, 1000), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=1234, shuffle=True, solver='sgd', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)
model = MLP.fit(x, y)


# Load test data and predict
vendor = pd.read_csv('test.txt', sep='\t')
v_smiles = vendor.iloc[:,1]
v_mols = [Chem.MolFromSmiles(s) for s in v_smiles]
v_mols_desc = np.array(describe(v_mols))

x2 = scaler2.transform(v_mols_desc)
x3 = pd.DataFrame(x2)
x4 = x3.dropna()
preds = MLP.predict(x4)
probs = MLP.predict_proba(x4)

nan = np.where(np.asanyarray(np.isnan(x3)))
nan_id = nan[0]
nan_id_unique = np.unique(nan_id)
nan_list = nan_id_unique.tolist()
vendor2 = vendor.drop(nan_list)

preds2 = pd.DataFrame(preds, columns=['Preds'])
probs2 = pd.DataFrame(probs, columns=['Prob_0', 'Prob_1'])
vendor2 = vendor2.reset_index()

final = pd.concat([vendor2, preds2, probs2], axis=1)
final.to_csv('test_preds.txt', sep='\t', index=False)