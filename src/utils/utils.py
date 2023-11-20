import random
import numpy as np
import math, csv
from random import sample


def assign_excess(interval, excess):
    ex = set()
    i = 0
    while i < excess:
        x = random.randint(0, interval - 1)
        if x in ex:
            continue
        else:
            ex.add(x)
            i += 1
    return ex


def intervals(data, interval):
    interval_list = [None] * interval
    sorted_data = np.sort(data)
    length = len(sorted_data)
    slice_range = math.floor(length / interval)  # To get a length of each interval
    excess = set()

    if slice_range * interval < length:
        excess = assign_excess(
            interval, length - slice_range * interval
        )  # Todo: Change this to binary operation

    # print(sorted_data)
    # print("Total Length: {}".format(length))
    # print("Slice Range: {}".format(slice_range))
    i = 0
    cnt = 0
    while i < length:
        e = 0
        if cnt in excess:  # Todo: Change this to binary operation
            e += 1
        if cnt == 0:
            partition = sorted_data[i - 1 + slice_range + e]
            interval_list[cnt] = ("neg_inf", sorted_data[i - 1 + slice_range + e])
            # print("{} {} = {}".format(sorted_data[i-1+slice_range+e],sorted_data[i+slice_range+e], partition))
        elif i + slice_range + e == length:
            interval_list[cnt] = (partition, "pos_inf")
        else:
            new_partition = (
                sorted_data[i - 1 + slice_range + e] + sorted_data[i + slice_range + e]
            ) / 2  # This will be passed to the next beginning of slice
            interval_list[cnt] = (partition, new_partition)
            partition = new_partition
            # print("{} {} = {}".format(sorted_data[i-1+slice_range+e],sorted_data[i+slice_range+e], new_partition))

        # print("{} - {}".format(interval_list[cnt][0],interval_list[cnt][1]))
        i += slice_range + e
        cnt += 1

    return interval_list


def generateRuleSets(N, all_intervals, n_class):
    rules = []

    for n in range(0, N):
        rules.append(generateRule(all_intervals, n_class))
    return rules


def generateRule(all_intervals, n_class):
    rule = []
    length = len(all_intervals)
    n_intervals = len(all_intervals[0])

    for n in range(0, length):
        rule.append(
            all_intervals[n][random.randint(0, n_intervals - 1)]
        )  # Interval Added

    for p in generateParam(n_class):
        rule.append(p)
    return rule


def generateParam(n_class):
    weight = []

    for i in range(0, n_class):
        weight.append(random.random())
    # print("Weight before softmax {}".format(weight))
    # print("parameter {}".format(softmax(weight)))
    return softmax(weight)


def generateBundles(n_bundles, n_rules, all_intervals, n_class):
    bundles = []
    for i in range(n_bundles):
        bundles.append(generateRuleSets(n_rules, all_intervals, n_class))
    return bundles


def check_rule(val, criteria):
    if criteria[0] == "neg_inf":
        # print("Value to check: {} -- Rule: val <= {} ---------- {}".format(val, criteria[1], val<=criteria[1]))
        if val < criteria[1]:
            return 1
    elif criteria[1] == "pos_inf":
        # print("Value to check: {} -- Rule: {} < = val ---------- {}".format(val, criteria[0], criteria[0]< val))
        if criteria[0] <= val:
            return 1
    else:
        # print("Value to check: {} -- Rule: {} <= val <= {} ---------- {}".format(val, criteria[0], criteria[1], criteria[0]<=val and val<=criteria[1]))
        if criteria[0] <= val and val < criteria[1]:
            return 1

    return 0


def isEqualArr(arr1, arr2):
    # Check if two arrays are identical
    if len(arr1) != len(arr2):
        return False
    else:
        for i, j in zip(arr1, arr2):
            if i != j:
                return False
    return True


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def evolve_bundle(
    bundles,
    accuracies,
    r_cross,
    r_mut,
    all_intervals,
    n_class,
    n_parents,
    n_children,
    rule_evolve,
    two_best=False,
):
    # Select) randomly multiple sets of rule bundles that predict the class - If none found by the end of "trial", return to 1
    # print("bundles before: len={} and {}".format(len(bundles), bundles))

    prob = getProbs(accuracies)
    # print("Accuracy {}".format(accuracies))
    # print("Probability {}".format(prob))

    # Select parents to crossover probablistically
    if two_best:
        evo_bundles = select_two_best_parents(bundles, accuracies, n_parents)
    else:
        evo_bundles = select_parents(bundles, prob, n_parents)

    children = []
    i = 0

    while i < n_children:
        # print("Counter{}".format(i))
        # Crossover)
        p1, p2 = select(evo_bundles, 2)
        child_bundle_a, child_bundle_b = crossover_bundle(p1, p2, rule_evolve, r_cross)

        # Mutate bundles)
        if random.random() < r_mut:
            print("Mutation")
            child_bundle_a = mutate_bundle(child_bundle_a, all_intervals, n_class)
        if random.random() < r_mut:
            print("Mutation")
            child_bundle_b = mutate_bundle(child_bundle_b, all_intervals, n_class)

        children.append(child_bundle_a)
        children.append(child_bundle_b)

        i += 2

    bundles = replace_child(bundles, children)
    # Pick parents to be replaced
    # survivors=select_replacement(bundles, n_children)
    # print("Length of Survivor {}".format(len(survivors)))

    # Concatenate
    # bundles=survivors+children
    # print("bundles after ': len={} and {}".format(len(bundles), bundles))
    return bundles


def replace_child(bundles, children):
    randomlist = random.sample(range(0, len(bundles)), len(children))
    k = 0
    for i in randomlist:
        bundles[i] = children[k]
        k += 1
    return bundles


def select(obj, n_sample):
    return sample(obj, n_sample)


# Select parents in accordance with accuracy scores
def select_parents(bundles, prob, n_parents):
    # evo=random.choices(bundles, weights=prob, k=n_parents)
    temp_bundles = np.array(bundles)
    idx = np.random.choice(
        np.arange(0, len(temp_bundles)), size=n_parents, replace=False, p=prob
    )
    # idx=np.random.choice(np.arange(0,len(temp_bundles)), size=n_parents, replace=False, p=prob)
    evo = np.array(bundles)[idx]
    return evo.tolist()


def select_two_best_parents(bundles, accuracies, n_parents):
    selection_ix = np.argpartition(accuracies, -2)[-2:]
    return [bundles[selection_ix[0]], bundles[selection_ix[1]]]


# mutation operator
def mutate_rule(child_rule, n_class, all_intervals):
    n_intervals = len(all_intervals[0])

    for i in range(0, len(child_rule) - n_class):
        child_rule[i] = all_intervals[i][random.randint(1, n_intervals - 1)]
    child_rule[-n_class:] = generateParam(n_class)

    # print("child_rule:{}".format(child_rule[-n_class:]))

    return child_rule


# mutation operator
def mutate_bundle(child_bundle, all_intervals, n_class):
    for child_rule in child_bundle:
        child_rule = mutate_rule(child_rule, n_class, all_intervals)
    return child_bundle


# crossover two parents to create two children bundles
def crossover_bundle(p1, p2, rule_evolve, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # select crossover point that is not on the end of the string
    pt = random.randint(1, len(c1) - 1)
    # perform crossover
    c1 = p1[:pt] + p2[pt:]
    c2 = p2[:pt] + p1[pt:]

    if rule_evolve and random.random() < r_cross:
        c1 = crossover_rules(c1)
        c2 = crossover_rules(c2)
    return c1, c2


def crossover_rules(p):
    if len(p) % 2 == 0:
        mid = len(p) // 2
        sep = mid
        c1 = p[:mid].copy()
        c2 = p[sep:].copy()
        c1, c2 = crossover(c1, c2)
        return c1 + c2

    else:
        mid = len(p) // 2
        sep = mid + 1
        c1 = p[:mid].copy()
        c2 = p[sep:].copy()
        c3 = [p[mid]]
        c1, c2 = crossover(c1, c2)
        return c1 + c3 + c2


def crossover(c1, c2):
    random.shuffle(c1)
    random.shuffle(c2)
    for r1, r2 in zip(c1, c2):
        # select crossover point that is not on the end of the string
        pt = random.randint(1, len(c1) - 1)
        # perform crossoverhy
        r1 = r1[:pt] + r2[pt:]
        r2 = r2[:pt] + r1[pt:]
    return c1, c2


def getProbs(accuracies):
    Ttl = sum(accuracies)
    if Ttl == 0:
        prob = [1 / len(accuracies) for i in accuracies]
        # print(prob)
    else:
        prob = [x / Ttl for x in accuracies]
        # print(prob)
    return prob


def predict(rule, n_class, X_data):
    pred = zerolistmaker(n_class)
    val = 0
    # print("Input Data:{}".format(X_data))
    for k in range(0, len(X_data)):
        val += check_rule(X_data[k], rule[k])
    if val == len(X_data):
        # print("----------Accepted---------- {}".format(val))
        # print("data {}".format(X_data))
        # print("rule {}".format(rule))
        pred = rule[-n_class:]
        # print("Increment {}".format(pred))
    # else:
    # print("----------rejected---------- {}".format(val))
    # print("data {}".format(X_data))
    # print("rule {}".format(rule))
    return pred


def convert_predictor(res):
    ans = zerolistmaker(len(res))
    # print("Original output {}".format(res))
    new_list = sorted(res, key=None, reverse=True)
    # print("Sorted List {}".format(new_list))
    first = new_list[0]
    second = new_list[1]
    if first != second:
        # idx=np.where(res==first)[0][0]
        idx = res.index(first)
        ans[idx] = 1
    return ans


def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros


def vote(bundle, n_class, X_data):
    vote = [0, 0, 0]
    for rule in bundle:
        # print("Vote before:{}".format(vote))
        pred = predict(rule, n_class, X_data)
        vote = sum_pred(vote, pred)
        # print("Vote After {}".format(vote))
    return vote


def sum_pred(list1, list2):
    sum_list = []
    for item1, item2 in zip(list1, list2):
        sum_list.append(item1 + item2)
    return sum_list


def eval_bundle(bundle, n_class, input, obj):
    score = 0
    # Loop over all the training data
    for i in range(0, len(input)):
        # Loop over the rule to examine its fitness
        # print("Bundle {} input {}".format(bundle, input.iloc[i]))
        v = convert_predictor(vote(bundle, n_class, input.iloc[i]))
        # print("{} vs {} --- {}".format(v, obj[i], isEqualArr(v, obj[i])))
        if isEqualArr(v, obj[i]):
            score += 1
    return score / len(obj)  # Return the % of accuracy for the target class


def bundle_to_csv(bundle, fname):
    filename = str(fname + ".csv")
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerows(bundle)


def import_bundle(fname):
    # open file in read mode
    bundle = []
    with open(fname, "r") as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = csv.reader(read_obj, skipinitialspace=True)
        # Iterate over each row in the csv using reader object

        for row in csv_reader:
            i = 0
            rule = []
            for p in row:
                if i > 3:
                    rule.append(float(p))
                else:
                    rule.append(eval(p))
                i += 1
            bundle.append(rule)
    return bundle
