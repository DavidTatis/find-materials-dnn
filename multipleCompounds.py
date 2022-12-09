# %%
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Lambda, BatchNormalization
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, Input,concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from tensorflow.keras.utils import plot_model
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import pickle
import timeit
from operator import itemgetter
# %%

# Model
def create_new_model():
    input_vec = Input(shape=(86,))
    x0=Activation('relu')(BatchNormalization()(Dense(1024)(input_vec)))
    m1=concatenate([input_vec, x0], axis=-1)
    x1=Activation('relu')(BatchNormalization()(Dense(1024)(m1)))
    m2=concatenate([m1, x1], axis=-1)
    x2=Activation('relu')(BatchNormalization()(Dense(1024)(m2)))
    m3=concatenate([m2, x2], axis=-1)
    x3=Activation('relu')(BatchNormalization()(Dense(1024)(m3)))

    m4=concatenate([m3, x3], axis=-1)
    x4=Activation('relu')(BatchNormalization()(Dense(512)(m4)))
    m5=concatenate([m4, x4], axis=-1)
    x5=Activation('relu')(BatchNormalization()(Dense(512)(m5)))
    m6=concatenate([m5, x5], axis=-1)
    x6=Activation('relu')(BatchNormalization()(Dense(512)(m6)))


    m7=concatenate([m6, x6], axis=-1)
    x7=Activation('relu')(BatchNormalization()(Dense(256)(m7)))
    m8=concatenate([m7, x7], axis=-1)
    x8=Activation('relu')(BatchNormalization()(Dense(256)(m8)))
    m9=concatenate([m8, x8], axis=-1)
    x9=Activation('relu')(BatchNormalization()(Dense(256)(m9)))

    m10=concatenate([m9, x9], axis=-1)
    x10=Activation('relu')(BatchNormalization()(Dense(128)(m10)))
    m11=concatenate([m10, x10], axis=-1)
    x11=Activation('relu')(BatchNormalization()(Dense(128)(m11)))
    m12=concatenate([m11, x11], axis=-1)
    x12=Activation('relu')(BatchNormalization()(Dense(128)(m12)))

    m13=concatenate([m12, x12], axis=-1)
    x13=Activation('relu')(BatchNormalization()(Dense(64)(m13)))
    m14=concatenate([m13, x13], axis=-1)
    x14=Activation('relu')(BatchNormalization()(Dense(64)(m14)))

    m15=concatenate([m14, x14], axis=-1)
    x15=Activation('relu')(BatchNormalization()(Dense(32)(m15)))

    m16=concatenate([m15, x15], axis=-1)
    x16=Dense(1,activation='linear')(m16)

    model=Model(input_vec,x16)
    # model.summary()
    return model

# %%

folder_path = "/root/Notebooks"
folder_path += "/new_materials"
##################################
# create files
##################################

f = open(folder_path +"/data.csv","w+")
f2 = open(folder_path +"/compositions.txt","w+")

##################################
# elements
##################################

elements = ["H","Li","Be","B","C","N","O","F","Na","Mg","Al",
			"Si","P","S","Cl","K","Ca","Sc","Ti","V","Cr","Mn",
			"Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
			"Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag",
			"Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce",
			"Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm",
			"Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
			"Tl","Pb","Bi","Ac","Th","Pa","U","Np","Pu"]


# Na, Cl, K3, P, O4,  Li, F
# Na2HPO4
# Na Cl Na2 H P O4 Li F

naIndex=elements.index("Na")
clIndex=elements.index("Cl")
kIndex=elements.index("K")
pIndex=elements.index("P")
oIndex=elements.index("O")
liIndex=elements.index("Li")
fIndex=elements.index("F")
print(naIndex,clIndex,kIndex,pIndex,oIndex,liIndex,fIndex)

atoms=[1,1,3,1,4,1,1]
formated={}
byAtoms={}
posibilities=[]

maxMol=50
population=[]
for x in range(1,maxMol):
	for y in range(1,maxMol):
		for z in range(1,10):
			temp=[atoms[0]*x,atoms[1]*x,atoms[2]*y,atoms[3]*y,atoms[4]*y,atoms[5]*z,atoms[6]*z]
			posibilities.insert(0,temp)
			ratios = [0]*len(elements)

			a=ratios[naIndex]=round(temp[0]/sum(temp),3)
			b=ratios[clIndex]=round(temp[1]/sum(temp),3)
			c=ratios[kIndex]=round(temp[2]/sum(temp),3)
			d=ratios[pIndex]=round(temp[3]/sum(temp),3)
			e=ratios[oIndex]=round(temp[4]/sum(temp),3)
			f=ratios[liIndex]=round(temp[5]/sum(temp),3)
			g=ratios[fIndex]=round(temp[6]/sum(temp),3)			
			if f"Na{a},Cl{b},K{c},P{d},O{e},Li{f},F{g}" not in formated:
				population.insert(len(population),ratios)
				formated[f"Na{a},Cl{b},K{c},P{d},O{e},Li{f},F{g}"] = len(population)-1 #save the index where is the compound  in the population
				byAtoms[f"Na{x}, Cl{x}, K{y}*3, P{y}, O{y}*4, Li{z}, F{z}"]=len(population)-1 #save the index where is the compound  in the population
				
				

# print(list(formated))
print("total predictions: ",len(formated))

model=create_new_model()
model_path="/root/data/IRNET-0.4.h5"
model.load_weights(model_path)

predictions=list(model.predict(np.asarray(population)))

sortIndex=sorted(range(len(predictions)), key=lambda k: predictions[k])
print(sortIndex[0:5])

print(f"{min(predictions)}:min prediction.  {max(predictions)}: max prediction.")

for top in range (10):
	print(f"======Predicted as Top {top}======")
	print(predictions[sortIndex[top]])
	formatedValue = {i for i in formated if formated[i]==sortIndex[top]}
	byAtomsValue={i for i in byAtoms if byAtoms[i]==sortIndex[top]}
	print("by ratio: ",formatedValue)
	# print("by atoms: ",byAtomsValue)
	print(f"{min(predictions)} = {byAtomsValue}")



# print(sortedIndex[0:10])
# print(np.asarray(predictions)[0:10])
# print("Eval. Population ",len(population))


# %%
