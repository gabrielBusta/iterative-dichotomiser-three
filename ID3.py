from math import log
from collections import defaultdict


def main():
    data = [
        ({'outlook':'sunny', 'temperature':'hot', 'humidity':'high', 'wind':'weak'}, False),
        ({'outlook':'sunny', 'temperature':'hot', 'humidity':'high', 'wind':'strong'}, False),
        ({'outlook':'overcast', 'temperature':'hot', 'humidity':'high', 'wind':'weak'}, True),
        ({'outlook':'rain', 'temperature':'mild', 'humidity':'high', 'wind':'weak'}, True),
        ({'outlook':'rain', 'temperature':'cool', 'humidity':'normal', 'wind':'weak'}, True),
        ({'outlook':'rain', 'temperature':'cool', 'humidity':'normal', 'wind':'strong'}, False),
        ({'outlook':'overcast', 'temperature':'cool','humidity':'normal', 'wind':'strong'}, True),
        ({'outlook':'sunny', 'temperature':'mild', 'humidity':'high',  'wind':'weak'},  False),
        ({'outlook':'sunny', 'temperature':'cool', 'humidity':'normal',  'wind':'weak'}, True),
        ({'outlook':'rain', 'temperature':'mild', 'humidity':'normal',  'wind':'weak'}, True),
        ({'outlook':'sunny', 'temperature':'mild', 'humidity':'normal', 'wind':'strong'}, True),
        ({'outlook':'overcast', 'temperature':'mild', 'humidity':'high', 'wind':'strong'}, True),
        ({'outlook':'overcast', 'temperature':'hot', 'humidity':'normal',  'wind':'weak'}, True),
        ({'outlook':'rain', 'temperature':'mild', 'humidity':'high', 'wind':'strong'}, False)
    ]

    # create a dictionary mapping each attribute A
    # to a list of all possible values V for A.
    # For example:
    # {'outlook':['sunny', 'overcast', ..], 'wind':['strong', 'weak', ..], ..}
    possible_values = defaultdict(list)
    for sample in data:
        for attr in sample[0]:
            if sample[0][attr] not in possible_values[attr]:
                possible_values[attr].append(sample[0][attr])

    root = decision_tree(data, possible_values)

    print_tree(root, depth=1)

    # DEMO

    sample = {'outlook':'sunny', 'temperature':'mild', 'humidity':'high', 'wind':'weak'}
    print('classifiying')
    print(sample)
    classification = classify(sample, root)
    print('category:', classification)

    sample = {'outlook':'rain', 'temperature':'hot', 'humidity':'normal', 'wind':'weak'}
    print('classifiying')
    print(sample)
    classification = classify(sample, root)
    print('category:', classification)


# the data in a node is the name of the attribute the node splits on
# unless the node is a leaf. if the node is a leaf then the data
# is a Boolean value indicating whether the node is positive or negative.
class Node:
    def __init__(self, content, isLeaf):
        self.content = content
        self.isLeaf = isLeaf
        self.edges = []
    def add_edge(self, edge):
        self.edges.append(edge)


# the tail is the parent. the head is the child. tail -> head
# the label of the edge represents a value of the attribute the tail splits on
class Edge:
    def __init__(self, tail, head, label):
        self.tail = tail
        self.head = head
        self.label = label


# this function recursively builds the decision tree based on the training data
def decision_tree(data, possible_values):
    # if the data is pure we've reached a leaf
    if entropy(data) == 0.0:
        return Node(leaf_value(data), isLeaf=True)

    node = Node(decision_attr(data, possible_values), isLeaf=False)
    for value in possible_values[node.content]:
        subset = select_samples(data, node.content, value)
        node.add_edge(Edge(node, decision_tree(subset, possible_values), value))

    return node


def classify(sample, node):
    # if the node is a leaf return True or False
    if node.isLeaf:
        return node.content

    # go down the edge corresponding to the sample's value
    # for the current decision attribute
    edge = next(e for e in node.edges if e.label == sample[node.content])
    classification = classify(sample, edge.head)

    return classification


# this function determines if a leaf node should be True or False
def leaf_value(data):
    pos = len([sample for sample in data if sample[1] == True])
    neg = len([sample for sample in data if sample[1] == False])
    if pos > neg:
        return True
    else:
        return False


# this function returns the attribute with the higest gain in a dataset
def decision_attr(data, possible_values):
    # build a dictionary of the form {attribute:gain}
    attr_gain = {}
    for attr in possible_values:
        attr_gain[attr] = gain(data, attr, possible_values)

    # return the attribute with the highest gain
    max_gain = max(attr_gain.values())
    return next(attr for attr, gain in attr_gain.items() if gain == max_gain)


# this function returns a list of  all the samples in the provided data set
# where the specified attribute has the desired value
def select_samples(data, attr, value):
    return [sample for sample in data if sample[0][attr] == value]


def gain(data, target_attr, possible_values):
    # this list contains the value of the terms required  to calculate the gain
    terms = []

    # the first term is entropy(S), i.e. the entropy of the entire sample set
    terms.append(entropy(data))

    # for each possible value, v, of the target attribute
    # calculate -( (|Sv| / |S|) * entropy(Sv) )
    # this yields the rest of the terms
    total = len(data) # |S|

    for value in possible_values[target_attr]:
        subset = select_samples(data, target_attr, value)
        terms.append(-(len(subset) / total)*(entropy(subset)))

    # sum all of the terms together to obtain the gain
    return sum(terms)


def entropy(data):
    total = len(data) # |S|

    # find the number of postive samples in the data set
    pos = len([sample for sample in data if sample[1] == True])

    # find the number of negative samples in the data set
    neg = len([sample for sample in data if sample[1] == False])

    # if the data is perfectly classified the entropy is 0.0
    if (pos == total or neg == total):
        return 0.0

    pos_p = pos / total
    neg_p = neg / total

    return -(pos_p * log(pos_p, 2)) - (neg_p * log(neg_p, 2))


def print_tree(node, depth):
    if node.isLeaf:
        return

    if depth == 1:
        print('root')
    else:
        print('level', depth)

    print('attribute:', node.content)

    for e in node.edges:
        print('|-' + e.label + '->' + str(e.head.content))

    depth += 1

    # print an empty line
    print('')

    for e in node.edges:
        print_tree(e.head, depth)


if __name__ == '__main__':
    main()
