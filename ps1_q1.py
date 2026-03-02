import numpy as np
from astropy.io import fits # You might need to pip install this
import pylab as plt # only needed for verification
import scipy
path_labels = "data/labels.fits"
allstar = fits.open(path_labels)
# the labels are in an enormous table in element [1] of this FITS file
labels = allstar[1].data
#plt.scatter(labels['TEFF'], labels['LOGG'], s=1)
#plt.xlim(6000, 3500)
#plt.ylim(5, 0)
# make a reasonable red-giant-branch sample
RGB = True
RGB = np.logical_and(RGB, labels['TEFF'] > 3500.)
RGB = np.logical_and(RGB, labels['TEFF'] < 5400.)
RGB = np.logical_and(RGB, labels['LOGG'] < 3.0)
RGB = np.logical_and(RGB, labels['LOGG'] > 0.0)
RGB = np.logical_and(RGB, labels['H'] < 10.5)
#print(np.sum(RGB))
# make a plot that an astronomer likes to see
RGB_labels = labels[RGB]
#plt.scatter(RGB_labels['TEFF'], RGB_labels['LOGG'], c=RGB_labels['FE_H'], s=1)
#plt.xlim(5400, 3500)
#plt.xlabel("effective temperature")
#plt.ylim(3., 0.)
#plt.ylabel("log10 surface gravity")
#plt.colorbar(label="metallicity")
# make train, validation, and test data sets
rng = np.random.default_rng(17)
N_RGB = len(RGB_labels)
N_train, N_valid, N_test = 1024, 256, 512
I = rng.permutation(N_RGB)
I_train = I[0:N_train]
I_valid = I[N_train:N_train+N_valid]
I_test = I[N_train+N_valid:N_train+N_valid+N_test]

train_labels = RGB_labels[I_train]
valid_labels = RGB_labels[I_valid]
test_labels = RGB_labels[I_test]
#print(len(train_labels), len(valid_labels), len(test_labels))

train_labels_logg = train_labels['LOGG']
print(train_labels_logg.shape) # (num_spectra, 1)
import numpy as np
from matplotlib import pyplot as plt
train_features = np.load('data/train_features.npy')
valid_features = np.load('data/valid_features.npy')
test_features = np.load('data/test_features.npy')
for i in range(10):
    plt.plot(train_features[i] + i)
print(train_features.shape) # (num_spectra, num_pixels) 

### LINEAR REGRESSION ###
def linear_regression(X, Y, group):
    X = np.column_stack([np.ones(len(X)), X])
    x, residuals, rank, sv = scipy.linalg.lstsq(X, Y)  ###(WHERE x IS THE "LEAST SQUARES SOLUTION")
    Y_pred = X @ x                                     ###(PRED FOR PREDICTION)
    rms_err = np.sqrt(np.mean((Y - Y_pred)**2))        ### ROOT MEAN SQUARE ERROR
    plt.scatter(Y, Y_pred, s=2, alpha=0.5)
    plt.savefig(f'{group}.plot.png')
    plt.close()                             ### SO THEY GO IN AS SEPERATE PLOTS!
linear_regression(train_features, train_labels_logg, "train")
linear_regression(valid_features, valid_labels['LOGG'], "validate")
linear_regression(test_features, test_labels['LOGG'], "test_data")

### K NEAREST NEIGHBOR !!! ###
from sklearn import neighbors as neigh
from sklearn.neighbors import NearestNeighbors

def horse(X_train, Y_train, X, Y, group, n_neigh, weight):                ### NEIGH
    knn = neigh.KNeighborsRegressor(n_neighbors = n_neigh, weights = weight)
    knn.fit(X_train,Y_train)
    Y_pred = knn.predict(X)
    rms_err = np.sqrt(np.mean((Y - Y_pred)**2))
    plt.scatter(Y, Y_pred, s=2, alpha=0.5)
    plt.savefig(f'{group}_knn.plot.png')
    plt.close() 
horse(train_features, train_labels_logg, valid_features, valid_labels['LOGG'], 
      "train+validate", 8, "uniform")       ### BAD BAD K - FIND BETTER WAY TO PICK IN Q2
horse(train_features, train_labels_logg, test_features, test_labels['LOGG'], 
      "train+test", 8, "uniform")

### MULTI LAYER PERCEPTRON ###
from sklearn.neural_network import MLPRegressor as mlpr
def mlp(X_train, Y_train, X, Y, group):
    regr = mlpr(random_state=1, max_iter=2000, tol=0.1) ### HYPER PARAMETERS FROM MLP-REGRESSOR DOCUMENTATION
    regr.fit(X_train, Y_train)                          ### PICK BETTER, MORE SUITABLE ONES IN Q2
    Y_pred = regr.predict(X)
    rms_err = np.sqrt(np.mean((Y - Y_pred)**2))
    plt.scatter(Y, Y_pred, s=2, alpha=0.5)
    plt.savefig(f'{group}_mlp.plot.png')
    plt.close() 
mlp(train_features, train_labels_logg, valid_features, valid_labels['LOGG'], 
      "train+validate")
mlp(train_features, train_labels_logg, test_features, test_labels['LOGG'], 
      "train+test")