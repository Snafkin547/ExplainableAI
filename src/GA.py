import numpy as np
from statistics import mean
import matplotlib.pyplot as plt


class GA:
    def __init__(
        self,
        n_bundles,
        n_rules,
        generations,
        r_cross,
        r_mut,
        n_class,
        n_train,
        n_parents=20,
        n_children=10,
        interval=3,
        rule_evolve=True,
        two_best=False,
        dampening=(0, 1, 100),
    ):
        self.n_bundles = n_bundles
        self.n_rules = n_rules
        self.generations = generations
        self.r_cross = r_cross
        self.r_mut = r_mut
        self.n_class = n_class
        self.n_train = n_train
        self.n_parents = n_parents
        self.n_children = n_children
        self.interval = interval
        self.rule_evolve = rule_evolve
        self.two_best = two_best
        self.dampening = dampening

        self.gen_score = {}
        self.gen_best_score = {}
        self.gen_bundle = {}

        self.best_bundle = None

    def run_model(self, X_train, y_train, X_test, y_test):
        if self.n_parents < 1:
            n_parents = round(
                self.n_parents * self.n_bundles
            )  # if parent's ratio is set %, it takes concrete # of parents

        n_attributes = len(X_train.columns)

        col0 = intervals(
            X_train.iloc[:, 0], self.interval
        )  # Todo: Program it to be dynamic
        col1 = intervals(X_train.iloc[:, 1], self.interval)
        col2 = intervals(X_train.iloc[:, 2], self.interval)
        col3 = intervals(X_train.iloc[:, 3], self.interval)
        all_intervals = [col0, col1, col2, col3]

        # Generate n bundles with n_rules
        bundles = generateBundles(
            self.n_bundles, self.n_rules, all_intervals, self.n_class
        )
        fname = "{}b_{}r_{}mutR_initial".format(
            self.n_bundles, self.n_rules, self.r_mut
        )
        # bundle_to_csv(bundles, fname)
        for gen in range(self.generations):
            print("------------Generation {}------------".format(gen))
            print("")
            # print(bundles)
            accuracies = []
            l = len(X_train)
            sample_idx = np.random.choice(l, self.n_train)

            for i in range(0, len(bundles)):
                accuracy = eval_bundle(
                    bundles[i],
                    self.n_class,
                    X_train.iloc[sample_idx, :],
                    y_train[sample_idx],
                )
                accuracies.append(accuracy)

            # Measure fitness of parents each gen
            # input=pd.DataFrame(X_test)
            test_accuracies = []
            for i in range(0, len(bundles)):
                test_accuracy = eval_bundle(bundles[i], self.n_class, X_test, y_test)
                test_accuracies.append(test_accuracy)

            interim_score = mean(test_accuracies)
            interim_best_score = max(test_accuracies)
            print("Interim Ave. Accuracy {}".format(interim_score))
            print("Interim Best Accuracy {}".format(interim_best_score))

            self.gen_score[gen] = interim_score
            self.gen_best_score[gen] = interim_best_score
            self.gen_bundle[gen] = bundles

            # Generate new generations
            if self.dampening[0] > 0:
                if gen < self.dampening[0]:
                    # print("Gen {} dampening {} gen<dampening[0] r_mut:{}".format(gen, dampening[0], r_mut))
                    bundles = evolve_bundle(
                        bundles,
                        accuracies,
                        self.r_cross,
                        self.r_mut,
                        all_intervals,
                        self.n_class,
                        self.n_parents,
                        self.n_children,
                        self.rule_evolve,
                        self.two_best,
                    )
                elif gen == self.dampening[0]:
                    new_r_mut = self.r_mut * self.dampening[1]
                    # print("Gen {} dampening {} gen==dampening[0] r_mut:{} new_r_mut:{}".format(gen, dampening[0], r_mut, new_r_mut))
                    bundles = evolve_bundle(
                        bundles,
                        accuracies,
                        self.r_cross,
                        new_r_mut,
                        all_intervals,
                        self.n_class,
                        self.n_parents,
                        self.n_children,
                        self.rule_evolve,
                        self.two_best,
                    )
                elif gen > self.dampening[0]:
                    if gen - self.dampening[0] == self.dampening[2]:
                        # print("Before:{}".format(new_r_mut))
                        new_r_mut = new_r_mut * self.dampening[1]
                        # print("Gen {} every {} after {} gen>dampening[0] r_mut:{} new_r_mut:{}".format(gen, dampening[2], dampening[0], r_mut, new_r_mut))
                    else:
                        # print("No dampening")
                        bundles = evolve_bundle(
                            bundles,
                            accuracies,
                            self.r_cross,
                            new_r_mut,
                            all_intervals,
                            self.n_class,
                            self.n_parents,
                            self.n_children,
                            self.rule_evolve,
                            self.two_best,
                        )
            else:
                # print("No dampening")
                bundles = evolve_bundle(
                    bundles,
                    accuracies,
                    self.r_cross,
                    self.r_mut,
                    all_intervals,
                    self.n_class,
                    self.n_parents,
                    self.n_children,
                    self.rule_evolve,
                    self.two_best,
                )

        final_ave_accuracy = interim_score
        final_best_accuracy = interim_best_score

        self.best_bundle = bundles[test_accuracies.index(final_best_accuracy)]
        print(type(self.best_bundle))
        # plt.figure(figsize=(5,5))
        f, ax = plt.subplots(1, figsize=(15, 10))
        plt.plot(
            list(self.gen_score.keys()),
            list(self.gen_score.values()),
            label="Ave Score",
        )
        plt.plot(
            list(self.gen_score.keys()),
            list(self.gen_best_score.values()),
            label="Best Score",
        )
        plt.xticks(np.arange(0, self.generations, step=10))
        plt.yticks(np.arange(0, 1.2, step=0.2))

        condition = "{} bundles {} rules each, {} parents generate {} children".format(
            self.n_bundles, self.n_rules, self.n_parents, self.n_children
        )
        plt.title(condition, pad=20)
        result_a = "Ave Accuracy {}".format(round(final_ave_accuracy, 2))
        result_b = "Best Bundle Accuracy {}".format(round(final_best_accuracy, 2))
        plt.text((self.generations - 1) / 2, final_ave_accuracy * 0.9, result_a)
        plt.text((self.generations - 1) / 2, final_best_accuracy * 1.05, result_b)
        plt.show()
        fname = "{}gen_{}b_{}r_{}mutR_final".format(
            self.generations, self.n_bundles, self.n_rules, self.r_mut
        )
        bundle_to_csv(self.best_bundle, fname)
        return self.best_bundle, final_best_accuracy
