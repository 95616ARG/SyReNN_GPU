"""Methods for analyzing the ACAS Xu network with planes.
"""
from typing import DefaultDict
from experiments.experiment import Experiment
from timeit import default_timer as timer
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import offsetbox
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import random, math
from PIL import Image
from pysyrenn import PlanesClassifier
from experiments.polar_image import PolarImage
from collections import defaultdict
from pysyrenn.frontend.network import Network
from tqdm import tqdm

matplotlib.use("Agg")

colors = [
    "violet",
    "lime",
    "brown",
    "gray",
    "black",
    "blue",
    "teal",
    "skyblue", #"orange",
    "red",
    "yellow"
]

label_names = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]

def T(v: np.ndarray):
    m = np.identity(v.shape[0])
    m[-1, 0:-1] = v[0:-1]
    return m

def T_inv(v: np.ndarray):
    m = np.identity(v.shape[0])
    m[-1, 0:-1] = -v[0:-1]
    return m

def ApplyTransformation(points, M):
    points = np.array(points)
    points_extended = np.insert(points, points.shape[-1], 1, axis=-1)
    points_transformed = np.dot(points_extended, M)
    return np.delete(points_transformed, points.shape[-1], axis=-1)

def AlignUntilAxis(v: np.ndarray, axis):
    v = np.array(v)
    n = v.shape[-1]
    v = np.insert(v, n, 1, axis=-1)

    # Move v to origin
    M = T_inv(v[0,:])
    v = np.dot(v, M)

    # Rotate v to X0.
    for c in tqdm(range(n-1, axis, -1), desc="(post) Aligning"):
        # print("A->B: folding dim", c)
        Mk = R(n+1, c-1, c, math.atan2(v[1, c], v[1, c-1]))
        M = np.dot(M, Mk)
        v = np.dot(v, Mk)

    return M

def R(dims: int, axis_from: int, axis_to: int, theta) -> np.ndarray:
    r = np.identity(dims)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    r[axis_from, axis_from] = cos_theta
    r[axis_to, axis_to] = cos_theta
    r[axis_from, axis_to] = -sin_theta
    r[axis_to, axis_from] = sin_theta
    return r

def RotatePlane(a, b, c):
    n = len(a)
    M = AlignUntilAxis([a, b], 0)
    aa, bb, cc = ApplyTransformation([a, b, c], M)

    cc = np.insert(cc, n, 1, axis=-1)
    for col in tqdm(range(n-1, 1, -1), desc="(post) Rotating"):
        Mk = R(n+1, col-1, col, math.atan2(cc[col], cc[col-1]))
        M = np.dot(M, Mk)
        cc = np.dot(cc, Mk)

    return M

class MNISTPlanesExperiment(Experiment):

    def __init__(self, directory_name):
        super().__init__(directory_name)
        self.last_network_type = None
        self.last_network = None

    def get_points_mnist(self, network, n_points = 3, seed=24):
        dataset = self.load_input_data("mnist_test" ,False)
        process = dataset["process"]
        inputs = dataset["raw_inputs"]
        labels = dataset["labels"]
        # print(np.array(inputs).shape)
        # print(np.array(labels).shape)

        network_name = "%s_relu_%d_%d" % ("mnist", 4, 100)
        network = self.load_network(network_name)

        random.seed(seed)
        np.random.seed(seed)

        indices = list(range(len(labels)))
        random.shuffle(indices)
        labels = labels[indices]
        inputs = inputs[indices]

        points = []
        point_labels = []
        for i, image in enumerate(inputs):
            if len(points) >= n_points:
                break
            processed_image = process(image)
            truth = labels[i]
            prediction = np.argmax(network.compute([processed_image])[0])
            if (truth == prediction):
                points.append(processed_image)
                point_labels.append(truth)

                # take two more misclassified
                for j in range(i+1, len(inputs)):
                    if len(points) >= n_points:
                        break
                    if not (labels[j] == truth):
                        continue
                    processed_image = process(inputs[j])
                    prediction = np.argmax(network.compute([processed_image])[0])
                    if not (truth == prediction):
                        points.append(processed_image)
                        point_labels.append(truth)

                if len(points) < n_points:
                    points = []
                    point_labels = []

        if len(points) != n_points:
            print("no enough points")
            assert(False)

        print("(pre) Picked points with labels", point_labels)

        return np.array(points), np.array(point_labels)

    def get_points(self, split="train", n_planes=1, only_correct_on=None, corruption="fog", seed=24):
        """Returns the desired dataset."""
        random.seed(seed)
        np.random.seed(seed)

        network_name = "%s_relu_%d_%d" % ("mnist", 3, 100)
        network = self.load_network(network_name)

        all_images = [
            np
            .load(f"external/mnist_c/{corruption}/{split}_images.npy")
            .reshape((-1, 28 * 28))
            for corruption in ("identity", "fog", "rotate")
        ]
        labels = np.load(f"external/mnist_c/identity/{split}_labels.npy")

        indices = list(range(len(labels)))
        random.shuffle(indices)
        labels = labels[indices]
        all_images = [images[indices] / 255. for images in all_images]

        identity_images = all_images[0]
        plane = [identity_images[0]]
        plane_labels = [labels[0]]
        for n, image in enumerate(identity_images):
            if plane_labels[0] == labels[n]:
                prediction = np.argmax(network.compute([image])[0])
                if prediction != labels[n]:
                    print("(pre) Picked",n,"with label",labels[n])
                    plane.append(image)
                    if (len(plane) >= 3):
                        return [plane], plane_labels

        assert (False)

        # planes = list(zip(*all_images))
        # if n_planes is not None:
        #     planes = planes[:n_planes]
        #     plane_labels = labels[:n_planes]

        return [plane], plane_labels

    def run_for_network(self, network_type, input_plane, seed, data_csv, device, dry_run=False, takeMisclassified=True):
        """Gets the plane classifications for a single network and stores them.
        """

        planes, labels, image_shape = input_plane
        network_name = "%s_relu_%s" % network_type
        print(f"\n(SyReNN) Running {network_name} on device {device} with seed {seed}")

        if self.last_network_type == network_type:
            network = self.last_network
        else:
            network = self.load_network(network_name)
            self.last_network_type = network_type
            self.last_network = network

        # print(network.layers)
        # import pdb; pdb.set_trace()

        classifier = PlanesClassifier(network, planes, preimages=False)

        classifier.partial_compute(include_post=False,
                                   compute_pre=False,
                                   fuse_classify=(not dry_run),
                                   post_process=False,
                                   device = device,
                                   dry_run = dry_run)
        upolytope = classifier.transformed_planes[0]
        timing = upolytope["timing"]
        fhat_size = upolytope["fhat_size"]
        print("(SyReNN) fhat size  :", fhat_size)
        print("(SyReNN) affine time:", timing.affine, "ms")
        print("(SyReNN) PWL time   :", timing.pwl, "ms")
        # print("t_preimage:", timing.preimage)
        # print("t_classify_argmax:", timing.classify_argmax)
        # print("t_classify_extract:", timing.classify_extract)
        # print("t_communication:", timing.classify_extract)

        self.write_csv(data_csv, {
            "network": network_name,
            "device": device,
            "seed": seed,
            "fhat_size": upolytope["fhat_size"],
            "split_scale": upolytope["split_scale"],
            "t_fc": timing.fc,
            "t_conv2d": timing.conv2d,
            "t_norm": timing.norm,
            "t_relu": timing.relu,
            "t_argmax": timing.argmax,
            "t_affine": timing.affine,
            "t_pwl": timing.pwl,
            "t_total": timing.total,
            "t_preimage": timing.preimage,
            "t_classify_argmax": timing.classify_argmax,
            "t_classify_extract": timing.classify_extract,
        })

        this_csv = self.begin_csv(
            f"{network_name}_device{device}_seed{seed}", [
            "network",
            "device",
            "seed",
            "fhat_size",
            "split_scale",
            "t_fc",
            "t_conv2d",
            "t_norm",
            "t_relu",
            "t_argmax",
            "t_affine",
            "t_pwl",
            "t_total",
            "t_preimage",
            "t_classify_argmax",
            "t_classify_extract",
            ])

        self.write_csv(this_csv, {
            "network": network_name,
            "device": device,
            "seed": seed,
            "fhat_size": upolytope["fhat_size"],
            "split_scale": upolytope["split_scale"],
            "t_fc": timing.fc,
            "t_conv2d": timing.conv2d,
            "t_norm": timing.norm,
            "t_relu": timing.relu,
            "t_argmax": timing.argmax,
            "t_affine": timing.affine,
            "t_pwl": timing.pwl,
            "t_total": timing.total,
            "t_preimage": timing.preimage,
            "t_classify_argmax": timing.classify_argmax,
            "t_classify_extract": timing.classify_extract,
        })

        if not dry_run:
            self.record_artifact(
                upolytope,
                "%s/%s/upolytope" % (network_name, seed), "pickle")

            self.record_artifact(
                image_shape,
                "%s/%s/image_shape" % (network_name, seed), "pickle")

    def run(self):
        """Run the transformer, split the plane, and save to disk.
        """
        data_csv = self.begin_csv("data", [
            "network",
            "device",
            "seed",
            "fhat_size",
            "split_scale",
            "t_fc",
            "t_conv2d",
            "t_norm",
            "t_relu",
            "t_argmax",
            "t_affine",
            "t_pwl",
            "t_total",
            "t_preimage",
            "t_classify_argmax",
            "t_classify_extract",
            ])
        self.record_artifact("data", "data", "csv")

        # input 1
        devices = [int(i) for i in input("Device to run (e.g., -1,0 for CPU and GPU 0): ").split(",")]

        # input 2
        dry_run = input("Dry run? [y/n]: ").lower()[0] == "y"
        # dry_run = True

        network_types = np.array([
            ("mnist", "3_100"),
            ("mnist", "9_200"),
            ("mnist", "6_500"),
            ("mnist", "4_1024"),
            ("mnist", "convsmall"),
            ("mnist", "convmedium"),
            # ("mnist", "convbig_diffai"),
        ])
        for i, (net, spec) in enumerate(network_types):
            print(f"\t{i}: {net}_relu_{spec}_model")

        # input 3
        choices = input("Models to run (e.g., 0,2,3 or * for all): ")
        print("choice is: ", choices)
        if choices == "*":
            networks_to_run = [tuple(i) for i in network_types]
        else:
            choices = [int(i) for i in choices.split(",")]
            networks_to_run = [tuple(i) for i in network_types[choices]]

        for i, (net, spec) in enumerate(networks_to_run):
            print(f"\t{i}: {net}_relu_{spec}_model")

        planes_config = {}

        # input 4
        seeds = [int(i) for i in input("Random seeds (e.g., 42,99): ").split(",")]
        for seed in seeds:
            planes, labels = self.get_points(n_planes=1, seed=seed)
            labels = [[label, label, label] for label in labels]
            image_shape = (28, 28)
            planes_config[seed] = (planes, labels, image_shape)

        for network_type in networks_to_run:
            for device in devices:
                for seed in seeds:
                    self.run_for_network(
                        network_type,
                        planes_config[seed],
                        seed, data_csv, device, dry_run=dry_run, takeMisclassified=True)

        if dry_run:
            quit()

    def analyze(self):
        """Writes plots from the transformed planes.
        """

        data_csv = self.read_artifact("data")

        for row in data_csv:
            # print(row)
            network_name, device, seed, time, fhat_size, split_scale = list(row.values())[:6]

            print(f"\n(post) Plotting `{network_name}_seed_{seed}.png`...")

            upolytope = self.read_artifact(
                "%s/%s/upolytope" % (network_name, seed))

            image_shape = self.read_artifact(
                "%s/%s/image_shape" % (network_name, seed))

            cached_M_path = f"mnist_seed_{seed}_M.npy"
            try:
                M = np.load(cached_M_path, allow_pickle=True)
                print(f"(post) Using cached `mnist_seed_{seed}_M.npy`")
            except:
                M = RotatePlane(upolytope["input_plane"][0], upolytope["input_plane"][1], upolytope["input_plane"][2])
                np.save(cached_M_path, M, allow_pickle=True)
                print(f"(post) Computed and cached M `mnist_seed_{seed}_M.npy`")

            input_plane_transformed = ApplyTransformation(upolytope["input_plane"], M)[:,:2]
            preimages_transformed = np.matmul(upolytope["combinations"], input_plane_transformed)

            highlights = upolytope["input_plane_idxes"]

            labels_of_vert = defaultdict(set)
            for p_idx, vpolytope in enumerate(upolytope["polytopes"]):
                for j, vertex_idx in enumerate(vpolytope):
                    labels_of_vert[vertex_idx].add(upolytope["labels"][p_idx])

            def collinear(p0, p1, p2):
                return abs(np.cross(p1-p0, p2-p0)) < 1e-3

            def onBoundary(plane, vertex):
                for i in range(len(plane)):
                    # print (i, ((i+1)%len(plane)))
                    a = plane[i]
                    b = plane[(i+1)%len(plane)]
                    if collinear(a, b, vertex):
                        return True
                return False

            for i, (vertex_idx, vertex_labels) in enumerate(tqdm(labels_of_vert.items(), desc="(post) Picking boundaries")):
                if ((len(vertex_labels) > 2) or
                    (len(vertex_labels) == 2 and onBoundary(input_plane_transformed, preimages_transformed[vertex_idx]))):
                    highlights.append(vertex_idx)

            # print("Ploting...")
            self.plot_polytopes(
                upolytope,
                preimages_transformed,
                highlights,
                image_shape,
                "%s_seed_%s" % (network_name, seed) )

        return True

    def plot_polytopes(self, upolytope, preimages_transformed, highlights, image_shape, figure_label=""):
        # polytopes = np.array(polytopes)
        # labels = np.array(labels)
        # highlights = np.array(highlights)

        plt.figure(frameon=False)
        plt.gca().axis('off')
        # plt.scatter(polytopes[:, :, 0], polytopes[:, :, 1], s = 0, color = labels[:])

        for p_idx, vpolytope in enumerate(tqdm(upolytope["polytopes"], desc="(post) Plotting polytopes")):
            t = plt.Polygon(
                    preimages_transformed[vpolytope],
                    facecolor = colors[upolytope["labels"][p_idx]],
                    alpha = 1,
                    zorder = 10,
                    lw = 10,
                    label = upolytope["labels"][p_idx])
            plt.gca().add_patch(t)

        # Reverse the list of highlights to make sure the three vertices are on the top.
        highlights.reverse()
        plt.scatter(preimages_transformed[highlights][:,0], preimages_transformed[highlights][:,1], s=30, color="black", zorder=20, marker="o", alpha=1)
        for highlight in tqdm(highlights, desc="(post) Plotting boundaries"):
            im = np.matmul(
                upolytope["combinations"][highlight],
                upolytope["input_plane"]
            ).reshape(image_shape)
            # im = upolytope["pre-images"][highlight].reshape(image_shape)
            oi = offsetbox.OffsetImage(im, zoom = 1)
            box = offsetbox.AnnotationBbox(oi, (preimages_transformed[highlight][0], preimages_transformed[highlight][1]), frameon=False)
            box.set_zorder(20)
            plt.gca().add_artist(box)

        # plt.show()
        legend_elements = [
            # Patch(facecolor=colors[label],
            Line2D([0], [0],
                # marker='o',
                color=colors[label],
                lw=8,
                label=label_names[label])
            for label in np.sort(np.unique(upolytope["labels"])) ]


        plt.legend(handles=legend_elements, loc="upper right", prop={'size': 14})
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
        plt.savefig(f"{self.directory}/{figure_label}.png" , dpi=200)

if __name__ == "__main__":
    label = input("Label for this set of MNIST experiments (can be empty): ")
    if not (label == ""):
        label = "_" + label
    exp_name = f"mnist_planes{label}"
    MNISTPlanesExperiment(exp_name).main()
