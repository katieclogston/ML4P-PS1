# ML4P-PS1
Question 1: 
linear regression plots: "train.plot.png", "validate.plot.png" and "test_data.plot.png"
knn plots: "train_knn.plot.png", "train+validate_knn.plot.png", and "train+test_knn.plot.png"
mlp plots: "train+validate_mlp.plot.png" and "train+test_mlp.plot.png"
Note: my hyperparameters for these are generally not super great

Question 2: 
plots: "train+validate_knn_2.plot.png" and "train+test_knn_2.plot.png", and "train+test_mlp.plot2.png"
Notes: I don't believe there were any hyperparameters within my linear regression I don't know if this means that I did something wrong but hopefully that's ok. Similarly, my k nearest neighbor didn't involve any RNG seeds I don't think. I could have missed something, or I might being doing it wrong. I did however try a few different RNG seeds for the MLP and got fairly similar answers. 
  Choice of hyperparameters: For the hyperparameters I experimented with, I generally chose the values which minimized root mean square error. However, my method for selecting potential values was often nowhere near comprehensive, as I decided to do something more simple which I could understand, though it was ultimately likely less effective and not as precise. generally I would just choose a range of possible values and see which worked better than others (for hyperparameters with numerical values). 

Question 3:
For my final project, I plan on using machine learning to some effect within the work I'm doing with Hogg on infrared excess in sunlike stars. The objects we're currently looking at were originally found using machine learning methods so I might utilize a similar process.  

Sources: 
https://numpy.org/doc/2.2/reference/generated/numpy.linalg.lstsq.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html
https://scikit-learn.org/stable/auto_examples/neighbors/plot_regression.html#sphx-glr-auto-examples-neighbors-plot-regression-py
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
https://towardsdatascience.com/choosing-the-right-number-of-neighbors-k-for-the-k-nearest-neighbors-knn-algorithm-fbc635279ec7/
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
https://numpy.org/doc/2.1/reference/generated/numpy.argmin.html
https://stackoverflow.com/questions/10565868/multi-layer-perceptron-mlp-architecture-criteria-for-choosing-number-of-hidde
and Claude.ai 
