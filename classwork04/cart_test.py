# Create class PJ_Cart_Tree that can be used like in example:
# == == == == == == == == == == == == == == == == == == == == == ==
from sklearn.datasets import load_iris

from homework import PJ_Cart_Tree

iris = load_iris()
tree = PJ_Cart_Tree(max_depth=4, acceptable_impurity=0.2)
tree.fit(iris.data, iris.target)
cl = tree.predict(iris.data[4])
print('Classified as {}'.format(iris.target_names[cl]))
# == == == == == == == == == == == == == == == == == == == == == ==
#
# Tree should be no deeper than max_depth parameter, however it
# can be more shallow if acceptable_impurity is reached on earlier node.
# Impurity should be calculated with Gini equation.
# Attached notebook contains an example of CART algorithm core functions.
