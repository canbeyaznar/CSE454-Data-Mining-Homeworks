# Can BEYAZNAR
# 161044038

import itertools
import math

# our fpnode for fptree
# simple tree node class
class FPNode(object):
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.link = None
        self.children = []

    def has_child(self, item):
        for node in self.children:
            if node.item == item:
                return True
        return False

    def get_child(self, item):
        for node in self.children:
            if node.item == item:
                return node
        return None

    def add_child(self, item):
        child = FPNode(item, 1, self)
        self.children.append(child)
        return child


class FPTree(object):
    # build tree for fp_growth
    def __init__(self, transactions, minSupCount, root, count):
        self.freq = self.find_frequency_itemsets(transactions, minSupCount)
        self.headers = self.build_headers(self.freq)
        self.root = self.build_tree(transactions, root,
                                    count, self.freq, self.headers)



    # find same or unique parameters and count them
    @staticmethod
    def find_frequency_itemsets(transactions, minSupCount):
        item_counts = {}
        # count items for each transactions
        for each_transaction in transactions:
            for each_item in each_transaction:
                if each_item not in item_counts:
                    item_counts[each_item] = 1
                else:
                    item_counts[each_item] += 1

        for each_key in list(item_counts.keys()):
            # if count in item_counts[each_key]  less than minSupCount
            if item_counts[each_key] < minSupCount:
                del item_counts[each_key] # delete item from item_counts array



        return item_counts

    # build header parameters,
    # find parameter names from frequency list
    @staticmethod
    def build_headers(freq):
        headers = {}
        for key in freq.keys():
            headers[key] = None
        return headers

    # build fp tree
    def build_tree(self, transactions, item, count, freq, headers):
        # set root
        root = FPNode(item, count, None)
        # insert tree nodes and sort each trans item
        for each_transaction in transactions:
            sorted_items = [x for x in each_transaction if x in freq]
            sorted_items.sort(key=lambda x: freq[x], reverse=True)
            if len(sorted_items) > 0: # if number of items bigger than 0, insert them to tree
                self.insert_tree(sorted_items, root, headers)
        # return tree
        return root

    def insert_tree(self, items, node, headers):
        first_item = items[0]
        child = node.get_child(first_item)
        if child is not None:
            child.count += 1
        else:
            # Add child
            child = node.add_child(first_item)

            # Link it to header
            if headers[first_item] is None:
                headers[first_item] = child
            else:
                current = headers[first_item]
                while current.link is not None:
                    current = current.link
                current.link = child

        # Call function recursively
        remaining_items = items[1:]
        size_remaining_items = len(remaining_items)
        if size_remaining_items > 0:
            self.insert_tree(remaining_items, child, headers)

    # we have to know single paths for fp_growth algorithm
    # returns true if node has single path
    def has_single_path(self, node):
        num_children = len(node.children)
        if num_children > 1:
            return False
        elif num_children == 0:
            return True
        else:
            return True and self.has_single_path(node.children[0])


    def mine_patterns(self, minSupCount):
        # (1) if Tree contains a single path P then
        if self.has_single_path(self.root):
            return self.generate_pattern_list()
        else:
            return self.get_patterns(self.mine_sub_trees(minSupCount))

    def get_patterns(self, patterns):
        root_item = self.root.item
        if root_item is not None:
            # We are in a conditional tree
            new_patterns = {}
            for key in patterns.keys():
                new_patterns[tuple(sorted(list(key) + [root_item]))] = patterns[key]

            return new_patterns
        return patterns

    def generate_pattern_list(self):
        pattern_list = {}
        items = self.freq.keys()

        # If we are in a conditional tree,
        # the root item is a pattern on its own.
        if self.root.item is None:
            root_item = []
        else:
            root_item = [self.root.item]
            pattern_list[tuple(root_item)] = self.root.count

        for i in range(1, len(items) + 1):
            # (2) for each combination (denoted as Beta) of the nodes in the path P
            # (3) generate pattern Beta U Alpha with support count = minimum support count of nodes in Beta
            for subset in itertools.combinations(items, i):
                pattern = tuple(sorted(list(subset) + root_item))
                pattern_list[pattern] = \
                    min([self.freq[x] for x in subset]) # traverse frequency list of subset and get min value

        return pattern_list

    def mine_sub_trees(self, minSupCount):
        patterns = {}
        sorted_freq = sorted(self.freq.keys(),
                             key=lambda x: self.freq[x])

        # Get items in tree in reverse order of occurrences.
        # (4) for each a_i in the header of Tree
        for item in sorted_freq:
            suffixes = []
            cond_tree_input = []
            node = self.headers[item]

            # Follow node links to get a list of
            # all occurrences of a certain item.
            while node is not None:
                suffixes.append(node)
                node = node.link

            # For each occurrence of the item,
            # trace the path back to the root node.
            for suffix in suffixes:
                freq = suffix.count
                path = []
                parent = suffix.parent

                while parent.parent is not None:
                    path.append(parent.item)
                    parent = parent.parent

                for i in range(freq):
                    cond_tree_input.append(path)

            # Construct subtree and mine patterns
            # (6) construct Beta’s conditional pattern base and then Beta’s conditional FP tree Tree_B ;
            subtree = FPTree(cond_tree_input, minSupCount,
                             item, self.freq[item])
            subtree_patterns = subtree.mine_patterns(minSupCount)

            # Insert subtree patterns into main patterns dictionary.
            for pattern in subtree_patterns.keys():
                if pattern in patterns:
                    patterns[pattern] += subtree_patterns[pattern]
                else:
                    patterns[pattern] = subtree_patterns[pattern]

        return patterns


def find_frequent_patterns(transactions, minSupCount):
    # create fp tree
    fptree = FPTree(transactions, minSupCount, None, None)
    print("-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-")
    print("Header Values: ")
    for each_header in fptree.headers:
        print(each_header)
    print("-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-")
    print("-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-")
    print("Item counts: ")
    for key, value in fptree.freq.items():
        print(key, " --> ", value)
    print("-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-")

    # mine patterns for the result
    return fptree.mine_patterns(minSupCount)


def load_csv_file(filename: str):

    file = open(filename, 'r')
    input_lines = file.readlines()
    file.close()
    counter = 0
    for line in input_lines:
        line = line.strip().rstrip(',') #split parameters in each line
        line = line.split(',')
        input_lines[counter] = line # assign the parameters to array
        counter += 1
    # return preprocessed dataset
    return input_lines

def print_patterns(pattern_list):
    for key, value in pattern_list.items():
        for each_item_in_key in key:
            print(each_item_in_key, end=" ")
        print(" --> ", value)

def main(args):
    data = load_csv_file(args[0])
    min_support_count = math.ceil(args[1] * len(data)) # get min_support_count
    patterns = find_frequent_patterns(data, min_support_count)

    print("-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-")
    print("Frequent patterns : ")
    print_patterns(patterns)
    print("-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-")

if __name__ == "__main__":
    # For this homework, I pick 2 different dataset
    # 'input.csv' has less items, There are more items in the 'Data.csv' dataset
    args = ["input.csv", 0.2]
    main(args)