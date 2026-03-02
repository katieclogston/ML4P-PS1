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
# FIND K VALUE #
def find_k(weights):
    k_values = [3, 5, 8, 10, 15, 20, 30, 50, 100, 300, 500, 1000]
    rmse_values = []
    for k in k_values:
        knn = neigh.KNeighborsRegressor(n_neighbors = k, weights = weights)
        knn.fit(train_features,train_labels_logg)
        Y_pred = knn.predict(valid_features)
        rms_err = np.sqrt(np.mean((valid_labels['LOGG'] - Y_pred)**2))
        rmse_values.append(rms_err)
    index_ultimate_k = np.argmin(rmse_values)
    ultimate_k = k_values[index_ultimate_k]
    best_rmse = min(rmse_values)
    print(f"best k {ultimate_k}")
    print(f"best rmse {best_rmse}")
    return ultimate_k, best_rmse
k_uni , rmse_uni = find_k("uniform")

# WEIGHTS BASED ON DISTANCE OR UNIFORM? #
k_dist , rmse_dist = find_k("distance")

if rmse_uni < rmse_dist:
    final_k = k_uni
    final_weights = "uniform"
    print(final_k)
    print(final_weights)
else:
    final_k = k_dist
    final_weights = "distance"
    print(final_k)
    print(final_weights)

# NOW DO IT! #
def horse(X_train, Y_train, X, Y, group):                ### NEIGH
    knn = neigh.KNeighborsRegressor(n_neighbors = final_k, weights = final_weights)
    knn.fit(X_train,Y_train)                    ### THIS IS ALL ALMOST DEF INEFFICIENT
    Y_pred = knn.predict(X)                     ### PROBABLY DON'T HAVE TO DO ALL OF IT AGAIN...
    rms_err = np.sqrt(np.mean((Y - Y_pred)**2))
    plt.scatter(Y, Y_pred, s=2, alpha=0.5)
    plt.savefig(f'{group}_knn_2.plot.png')
    plt.close() 

horse(train_features, train_labels_logg, test_features, test_labels['LOGG'], 
      "train+test")

### MULTI LAYER PERCEPTRON ###
from sklearn.neural_network import MLPRegressor as mlpr
# EXPERIMENT WITH ARCHITECTURE #
potential_lans = [(10,), (50,), (100,),          ### POTENTIAL LAYER AND NODE NUMBERS ###
                  (200,), (100, 50), (200, 100),
                  (200, 100, 50), (300, 150, 75),
                  (100, 100), (50, 50, 50), (10, 10, 10, 10)]
lans_rmse = []
for lans in potential_lans:
    regr = mlpr(hidden_layer_sizes = lans, max_iter = 1000, random_state = 17) 
    regr.fit(train_features,train_labels_logg)                          
    Y_pred = regr.predict(valid_features)
    rms_err = np.sqrt(np.mean((valid_labels['LOGG'] - Y_pred)**2))
    lans_rmse.append(rms_err)

best_lans = potential_lans[np.argmin(lans_rmse)]

# EXPERIMENT WITH ACTIVATION #
activs = ['identity', 'logistic', 'tanh', 'relu']
activs_rmse = []
for activations in activs:
    regr = mlpr(hidden_layer_sizes = best_lans, activation = activations,
                max_iter = 1000, random_state = 17) 
    regr.fit(train_features,train_labels_logg)                          
    Y_pred = regr.predict(valid_features)
    rms_err = np.sqrt(np.mean((valid_labels['LOGG'] - Y_pred)**2))
    activs_rmse.append(rms_err)
best_activs = activs[np.argmin(activs_rmse)]

def mlp(X_train, Y_train, X, Y, group):
    for seed in [17, 42, 123, 456, 789]:
        regr = mlpr(hidden_layer_sizes=best_lans, activation = best_activs,
                max_iter = 1000, random_state = seed)  ### SHOULD PROB LOOK MORE INTO OTHER BUILT IN PARAMETERS...
        regr.fit(X_train, Y_train)                          
        Y_pred = regr.predict(X)
        rms_err = np.sqrt(np.mean((Y - Y_pred)**2))
        plt.scatter(Y, Y_pred, s=2, alpha=0.5)
        plt.savefig(f'{group}_mlp.plot2.png')
        plt.close() 
        print(rms_err)

mlp(train_features, train_labels_logg, test_features, test_labels['LOGG'], 
      "train+test")
