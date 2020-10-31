# MIT 6.034 Lab 6: k-Nearest Neighbors and Identification Trees
# Written by 6.034 Staff

from api import *
from data import *
import math
log2 = lambda x: math.log(x, 2)
INF = float('inf')


################################################################################
############################# IDENTIFICATION TREES #############################
################################################################################


#### Part 1A: Classifying points ###############################################

def id_tree_classify_point(point, id_tree):
    """Uses the input ID tree (an IdentificationTreeNode) to classify the point.
    Returns the point's classification.
    
    Apply_classifer to get next node, and pass in recursively
    Else at Leaf Node and return the classification of node
    """

    try:
        return id_tree_classify_point(point, id_tree.apply_classifier(point))
    except:
        return (id_tree.get_node_classification())


#### Part 1B: Splitting data with a classifier #################################

def split_on_classifier(data, classifier):
    """Given a set of data (as a list of points) and a Classifier object, uses
    the classifier to partition the data.  Returns a dict mapping each feature
    values to a list of points that have that value.
    
    Iterate through data using classifer to build a dictionary
    """
    classification = {}
    for point in data:
        key = classifier.classify(point)
        if key in classification:
            classification[key].append(point)
        else:
            classification[key] = [point]
    return classification


#### Part 1C: Calculating disorder #############################################

def branch_disorder(data, target_classifier):
    """Given a list of points representing a single branch and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the branch.

    Use formula from specification
    """
    classifications = split_on_classifier(data, target_classifier)
    nb = len(data)
    disorder = 0
    for classification in classifications:
        nbc = len(classifications[classification])
        disorder += (-1*nbc/nb) * log2(nbc/nb)
    return disorder

def average_test_disorder(data, test_classifier, target_classifier):
    """Given a list of points, a feature-test Classifier, and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the feature-test stump.
    Algo: http://web.mit.edu/6.034/wwwbob/knn+idtree-notes.pdf
    """
    nt = len(data)
    disorder = 0
    test = split_on_classifier(data, test_classifier)
    for branch in test:
        nb = len(test[branch])
        disorder += (nb/nt) * branch_disorder(test[branch], target_classifier)
    return disorder

## To use your functions to solve part A2 of the "Identification of Trees"
## problem from 2014 Q2, uncomment the lines below and run lab6.py:

# for classifier in tree_classifiers:
#     print(classifier.name, average_test_disorder(tree_data, classifier, feature_test("tree_type")))


#### Part 1D: Constructing an ID tree ##########################################

def find_best_classifier(data, possible_classifiers, target_classifier):
    """Given a list of points, a list of possible Classifiers to use as tests,
    and a Classifier for determining the true classification of each point,
    finds and returns the classifier with the lowest disorder.  Breaks ties by
    preferring classifiers that appear earlier in the list.  If the best
    classifier has only one branch, raises NoGoodClassifiersError.
    
    Use min and key function
    Check for one branch by using split_on_classifier
    """
    best_classifier = min(possible_classifiers, key=lambda x: average_test_disorder(data, x, target_classifier))
    if len(split_on_classifier(data, best_classifier)) == 1:
        raise NoGoodClassifiersError("Classifier has only one branch")
    else:
        return best_classifier

## To find the best classifier from 2014 Q2, Part A, uncomment:
# print(find_best_classifier(tree_data, tree_classifiers, feature_test("tree_type")))

def construct_greedy_id_tree(data, possible_classifiers, target_classifier, id_tree_node=None):
    """Given a list of points, a list of possible Classifiers to use as tests,
    a Classifier for determining the true classification of each point, and
    optionally a partially completed ID tree, returns a completed ID tree by
    adding classifiers and classifications until either perfect classification
    has been achieved, or there are no good classifiers left."""

    # initialize the tree
    if id_tree_node == None:
        id_tree_node = IdentificationTreeNode(target_classifier)

    # check if homogeneous (case 1) 
    split = split_on_classifier(data, target_classifier)
    if len(split) == 1:
        id_tree_node.set_node_classification(target_classifier.classify(data[0]))
        return id_tree_node
    
    # check for best classifier, or return tree (case 3)
    try:
        best_classifier = find_best_classifier(data, possible_classifiers, target_classifier)
    except:
        return id_tree_node

    # act on best classifier (case 2)
    possible_classifiers.remove(best_classifier)
    split = split_on_classifier(data, best_classifier)

    id_tree_node.set_classifier_and_expand(best_classifier, split)
    branches = id_tree_node.get_branches()
    for branch in branches:
        construct_greedy_id_tree(split[branch], possible_classifiers, target_classifier, branches[branch])

    return id_tree_node


## To construct an ID tree for 2014 Q2, Part A:
# print(construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type")))

## To use your ID tree to identify a mystery tree (2014 Q2, Part A4):
# tree_tree = construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type"))
# print(id_tree_classify_point(tree_test_point, tree_tree))

## To construct an ID tree for 2012 Q2 (Angels) or 2013 Q3 (numeric ID trees):
# print(construct_greedy_id_tree(angel_data, angel_classifiers, feature_test("Classification")))
# print(construct_greedy_id_tree(numeric_data, numeric_classifiers, feature_test("class")))


#### Part 1E: Multiple choice ##################################################

ANSWER_1 = "bark_texture"
ANSWER_2 = "leaf_shape"
ANSWER_3 = "orange_foliage"

ANSWER_4 = [2,3] # draw path
ANSWER_5 = [3] # will attempt to use all branches
ANSWER_6 = [2] # random noise C should not be included
ANSWER_7 = 2 # simpliest

ANSWER_8 = "No"
ANSWER_9 = "No"


#### OPTIONAL: Construct an ID tree with medical data ##########################

## Set this to True if you'd like to do this part of the lab
DO_OPTIONAL_SECTION = False

if DO_OPTIONAL_SECTION:
    from parse import *
    medical_id_tree = construct_greedy_id_tree(heart_training_data, heart_classifiers, heart_target_classifier_discrete)


################################################################################
############################# k-NEAREST NEIGHBORS ##############################
################################################################################

#### Part 2A: Drawing Boundaries ###############################################

BOUNDARY_ANS_1 = 3 # could partition by x and y or groups
BOUNDARY_ANS_2 = 4 # can't split by x or y since diagonal, and middle region is closer to B

BOUNDARY_ANS_3 = 1 # partition by x and y, but A should be merged
BOUNDARY_ANS_4 = 2 # A is merged, but can't split by x and y

BOUNDARY_ANS_5 = 2 # cluster by promixity
BOUNDARY_ANS_6 = 4 # no threshold or promixity
BOUNDARY_ANS_7 = 1 # not right cluster 
BOUNDARY_ANS_8 = 4 # no division should be here
BOUNDARY_ANS_9 = 4 # similar to 8, no division between As

BOUNDARY_ANS_10 = 4 # no perfect division
BOUNDARY_ANS_11 = 2 # cluster
BOUNDARY_ANS_12 = 1 # straight lines (no cluster)
BOUNDARY_ANS_13 = 4 # can't do either
BOUNDARY_ANS_14 = 4 # can't do either


#### Part 2B: Distance metrics #################################################

def dot_product(u, v):
    """Computes dot product of two vectors u and v, each represented as a tuple
    or list of coordinates.  Assume the two vectors are the same length."""
    return sum([i[0]*i[1] for i in list(zip(u,v))])


def norm(v):
    "Computes length of a vector v, represented as a tuple or list of coords."
    # square root of the dot product of itself
    return math.sqrt(dot_product(v,v))

def euclidean_distance(point1, point2):
    "Given two Points, computes and returns the Euclidean distance between them."
    return math.sqrt(sum([(i[0]-i[1])**2 for i in list(zip(point1,point2))]))

def manhattan_distance(point1, point2):
    "Given two Points, computes and returns the Manhattan distance between them."
    return sum([abs(i[0]-i[1]) for i in list(zip(point1,point2))])

def hamming_distance(point1, point2):
    "Given two Points, computes and returns the Hamming distance between them."
    return sum([0 if (i[0] == i[1]) else 1 for i in list(zip(point1,point2))])

def cosine_distance(point1, point2):
    """Given two Points, computes and returns the cosine distance between them,
    where cosine distance is defined as 1-cos(angle_between(point1, point2))."""
    num = dot_product(point1, point2)
    den = norm(point1) * norm(point2)
    return 1 - num/den


#### Part 2C: Classifying points ###############################################

def get_k_closest_points(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns a list containing the k points
    from the data that are closest to the test point, according to the distance
    metric.  Breaks ties lexicographically by coordinates."""

    # sort by coords for tie-breaker, then distance, then top k elements
    data.sort(key=lambda x: x.coords)
    data.sort(key=lambda x: distance_metric(point.coords, x.coords))
    return data[:k]

def knn_classify_point(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns the classification of the test
    point based on its k nearest neighbors, as determined by the distance metric.
    Assumes there are no ties."""

    k_closest_points = get_k_closest_points(point, data, k, distance_metric)
    classifications = [point.classification for point in k_closest_points]
    return max(set(classifications), key = classifications.count) 


## To run your classify function on the k-nearest neighbors problem from 2014 Q2
## part B2, uncomment the line below and try different values of k:
# print(knn_classify_point(knn_tree_test_point, knn_tree_data, 1, euclidean_distance))


#### Part 2C: Choosing k #######################################################

def cross_validate(data, k, distance_metric):
    """Given a list of points (the data), an int 0 < k <= len(data), and a
    distance metric (a function), performs leave-one-out cross-validation.
    Return the fraction of points classified correctly, as a float."""
    raise NotImplementedError

def find_best_k_and_metric(data):
    """Given a list of points (the data), uses leave-one-out cross-validation to
    determine the best value of k and distance_metric, choosing from among the
    four distance metrics defined above.  Returns a tuple (k, distance_metric),
    where k is an int and distance_metric is a function."""
    raise NotImplementedError


## To find the best k and distance metric for 2014 Q2, part B, uncomment:
# print(find_best_k_and_metric(knn_tree_data))


#### Part 2E: More multiple choice #############################################

kNN_ANSWER_1 = None
kNN_ANSWER_2 = None
kNN_ANSWER_3 = None

kNN_ANSWER_4 = None
kNN_ANSWER_5 = None
kNN_ANSWER_6 = None
kNN_ANSWER_7 = None


#### SURVEY ####################################################################

NAME = None
COLLABORATORS = None
HOW_MANY_HOURS_THIS_LAB_TOOK = None
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
