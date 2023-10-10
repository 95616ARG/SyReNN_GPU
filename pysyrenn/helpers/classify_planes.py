"""Methods using SyReNN to understand classification of a network on a plane.
"""
import numpy as np
from pysyrenn.frontend.network import Network
from pysyrenn.frontend.argmax_layer import ArgMaxLayer

from timeit import default_timer as timer

class PlanesClassifier:
    """Handles classifying a set of planes using SyReNN.
    """
    def __init__(self, network, planes, preimages=True):
        """Creates a new PlanesClassifier for the given @network and @planes.

        @planes should be a list of Numpy arrays with each one representing a
        V-representation polytope with (n_vertices, n_dims). If preimages=True
        is set, preimages of the endpoints of each classification region will
        be returned (otherwise, only the combinations will be).
        """
        self.network = network
        self.planes = planes
        self.preimages = preimages

        self.partially_computed = False
        self.transformed_planes = None
        self.timing = None

        self.computed = False
        self.classifications = None

    def partial_compute(
            self,
            include_post=True,
            compute_pre=False,
            fuse_classify=False,
            post_process=True,
            device= -1,
            dry_run=False):
        """Computes the relevant ExactLine and stores it for analysis.
        """
        if self.partially_computed:
            return

        self.transformed_planes = self.network.transform_planes(self.planes,
                                                                self.preimages,
                                                                include_post=include_post,
                                                                compute_pre=compute_pre,
                                                                fuse_classify=fuse_classify,
                                                                post_process=post_process,
                                                                device = device,
                                                                dry_run = dry_run)

        if dry_run:
            return

        if post_process:
            self.timing = [timing for _, timing in self.transformed_planes]
            self.transformed_planes = [plane for plane, _ in self.transformed_planes]

        self.partially_computed = True
        if fuse_classify and post_process:
            self.classifications = []
            for transformed_plane in self.transformed_planes:
                if include_post:
                    preimages = [preimage for preimage, _, _ in transformed_plane]
                    labels = [label for _, _, label in transformed_plane]
                else:
                    preimages = [preimage for preimage, _ in transformed_plane]
                    labels = [label for _, label in transformed_plane]
                self.classifications.append((preimages, labels))
            self.computed = True

    @classmethod
    def from_syrenn(cls, transformed_planes):
        """Constructs a partially-computed PlanesClassifier from ExactLines.
        """
        self = cls(None, None, None)
        self.transformed_planes = transformed_planes
        self.partially_computed = True
        return self

    def compute(self):
        """Returns the classification regions of network restricted to @planes.

        Returns a list with one tuple (pre_regions, corresponding_labels) for
        each plane in self.planes. pre_regions is a list of Numpy arrays, each
        one representing a VPolytope.

        In contrast to LinesClassifier, no attempt is made here to return the
        minimal set.
        """
        if self.computed:
            return self.classifications

        self.partial_compute()

        self.classifications = []
        classify_network = Network([ArgMaxLayer()])
        syrenn_server_time = 0
        post_compute_time = 0
        for upolytope in self.transformed_planes:
            pre_polytopes = []
            labels = []
            # First, we take each of the linear partitions and split them where
            # the ArgMax changes.
            postimages = [post for pre, post in upolytope]

            # start = timer()
            classified_posts = classify_network.transform_planes(
                postimages, compute_preimages=False, include_post=True)
            # syrenn_server_time += timer() - start

            start = timer()
            for vpolytope, classify_upolytope in zip(upolytope,
                                                     classified_posts):
                pre, post = vpolytope
                for combinations, classified_post in classify_upolytope:
                    pre_polytopes.append(np.matmul(combinations, pre))
                    mean_combination = np.mean(combinations, axis=0)

                    """
                    if (combinations.shape[0] != combinations.shape[1]):
                        print("\nclassified prepost:", post.shape, "\n", post)
                        print("\nclassified comb:", combinations.shape, "\n", combinations)
                        print("\nclassified post:", classified_post.shape, "\n", classified_post)

                        print("\nmean classified comb:", mean_combination.shape)
                        print(mean_combination)

                        print("\nmean classified comb * prepost:", np.matmul(mean_combination, post).shape)
                        print(np.matmul(mean_combination, post))
                        quit()
                    """

                    class_region_label = np.argmax(
                        np.matmul(mean_combination, post).flatten())
                    labels.append(class_region_label)
            self.classifications.append((pre_polytopes, labels))

            post_compute_time += timer() - start

        print("Post compute  time in classification: ", post_compute_time * 1000, "ms");
        # print("SyReNN Server time in classification: ", syrenn_server_time * 1000, "ms");

        self.computed = True
        return self.classifications
