"""Experiment to generate points that violate and satisfy ACAS Xu network properties with SyReNN."""
import numpy as np
from experiments.experiment import Experiment
import warnings
from timeit import default_timer as timer
import sys

class ACASPoints(Experiment):
    """An experiment to generate points that satisfy or do not satisfy a given ACAS property.
    Which of the 45 ACAS Xu networks and which property are specified by user input.
    """
    def run(self):
        self.seed = 24
        np.random.default_rng(self.seed)
        self.timeout = 600
        self.network = self.load_network(self.net)

        # Load pre-processing function for ACAS points
        input_helpers = self.load_input_data("acas")
        process = input_helpers["process"]

        property_false, property_true = self.property_regions(process, n_samples=self.num_points)
        if property_false is None and property_true is None:
            return

        np.save(f'{self.path}/N{self.net[-3]}{self.net[-1]}_property{self.prop}_false.npy', property_false[:self.num_points])
        np.save(f'{self.path}/N{self.net[-3]}{self.net[-1]}_property{self.prop}_true.npy', property_true[:self.num_points])


    def property_regions(self, process, n_samples=20):
        """Determine the sampling of the input space based on the ACAS property."""
        LB = [0.0, -np.pi, -np.pi, 100.0, 0.0]
        UB = [60760.0, np.pi, np.pi, 1200.0, 1200.0]
        if self.prop == 1 or self.prop==2:
            # ρ ≥ 55947.691, vown ≥ 1145, vint ≤ 60
            return self.find_regions(
                # These are sampled:
                (LB[2], UB[2]), (1145, UB[3]), (LB[4], 60),
                # These are SyReNN'd:
                (55947.691, UB[0]), (LB[1], UB[1]),
                process, n_samples=n_samples)
        elif self.prop == 3:
            # 1500 ≤ ρ ≤ 1800, −0.06 ≤ θ ≤ 0.06, ψ ≥ 3.10, vown ≥ 980, vint ≥ 960.
            return self.find_regions(
                (3.10, UB[2]), (980, UB[3]), (960, UB[4]),
                (1500, 1800), (-0.06, 0.06),
                process, n_samples=n_samples)  
        elif self.prop == 4:
            # 1500 ≤ ρ ≤ 1800, −0.06 ≤ θ ≤ 0.06, ψ = 0, vown ≥ 1000, 
            # 700 ≤ vint ≤ 800
            return self.find_regions(
                (0, 0), (1000, UB[3]), (760, 800),
                (1500, 1800), (-0.06, 0.06),
                process, n_samples=n_samples)   
        elif self.prop == 5:
            # 250 ≤ ρ ≤ 400, 0.2 ≤ θ ≤ 0.4, −3.141592 ≤ ψ ≤ −3.141592 + 0.005, 
            # 100 ≤ vown ≤ 400, 0 ≤ vint ≤ 400
            return self.find_regions(
                (LB[2], np.pi + 0.005), (LB[3], 400), (LB[4], 400),
                (250, 400), (0.2, 0.4),
                process, n_samples=n_samples)
        elif self.prop == 6:
            # 12000 ≤ ρ ≤ 62000, (0.7 ≤ θ ≤ 3.141592) ∨ (−3.141592 ≤ θ ≤ −0.7), 
            # −3.141592 ≤ ψ ≤ −3.141592 + 0.005, 100 ≤ vown ≤ 1200, 
            # 0 ≤ vint ≤ 1200
            cexs, points = self.find_regions(
                (LB[2], np.pi + 0.005), (LB[3], UB[3]), (LB[4], UB[4]),
                (12000, 62000), (0.7, UB[1]),
                process, n_samples=n_samples/2) 
            cexs2, points2 = self.find_regions(
                (LB[2], np.pi + 0.005), (LB[3], UB[3]), (LB[4], UB[4]),
                (12000, 62000), (LB[1], -0.7),
                process, n_samples=n_samples/2) 
            cexs.extend(cexs2)
            points.extend(points2)
            return cexs, points
        elif self.prop == 7:
            # 0 ≤ ρ ≤ 60760, −3.141592 ≤ θ ≤ 3.141592, −3.141592 ≤ ψ ≤ 3.141592, 
            # 100 ≤ vown ≤ 1200, 0 ≤ vint ≤ 1200
            return self.find_regions(
                (LB[2], UB[2]), (LB[3], UB[3]), (LB[4], UB[4]),
                (LB[0], UB[0]), (LB[1], UB[1]),
                process, n_samples=n_samples)
        elif self.prop == 8:
            # 0 ≤ ρ ≤ 60760, −3.141592 ≤ θ ≤ −0.75·3.141592, −0.1 ≤ ψ ≤ 0.1, 
            # 600 ≤ vown ≤ 1200, 600 ≤ vint ≤ 1200
            return self.find_regions(
                (-0.1, 0.1), (600, UB[3]), (600, UB[4]),
                (LB[0], UB[0]), (LB[1], -0.75 * np.pi),
                process, n_samples=n_samples)
        elif self.prop == 9:
            #  2000 ≤ ρ ≤ 7000, −0.4 ≤ θ ≤ −0.14, 
            # −3.141592 ≤ ψ ≤ −3.141592 + 0.01, 100 ≤ vown ≤ 150, 0 ≤ vint ≤ 150
            return self.find_regions(
                (LB[2], -np.pi + 0.01), (LB[3], 150), (LB[4], 150),
                (2000, 7000), (-0.4, -0.14),
                process, n_samples=n_samples)
        elif self.prop == 10:
            # 36000 ≤ ρ ≤ 60760, 0.7 ≤ θ ≤ 3.141592, 
            # −3.141592 ≤ ψ ≤ −3.141592 + 0.01, 900 ≤ vown ≤ 1200, 600 ≤ vint ≤ 1200
            return self.find_regions(
                (LB[2], -np.pi + 0.01), (900, UB[3]), (600, UB[4]),
                (36000, UB[0]), (0.7, UB[1]),
                process, n_samples=n_samples)
        else:
            raise Exception(f"Property {self.prop} not defined.")

    def valid_network(self):
        """Not all ACAS properties apply to all networks.
        This function verifies that the specified property applies to the network.
        """
        aprev = int(self.net[-3])
        tau = int(self.net[-1])
        assert(aprev in [*range(1, 6)])
        assert(tau in [*range(1, 10)])

        if self.prop == 1:
            warnings.warn("Unlikely to find violation of this property.")
            return True
        elif self.prop == 2:
            return aprev >= 2
        elif self.prop == 3 or self.prop == 4:
            warnings.warn("Unlikely to find violation of this property.")
            return (aprev != 1 or tau not in [7, 8, 9])
        elif self.prop == 5 or self.prop == 6:
            return aprev == 1 and tau == 1
        elif self.prop == 7:
            return aprev == 1 and tau == 9
        elif self.prop == 8:
            return aprev == 2 and tau == 9
        elif self.prop == 9:
            warnings.warn("Unlikely to find violation of this property.")
            return aprev == 3 and tau == 3
        elif self.prop == 10:
            warnings.warn("Unlikely to find violation of this property.")
            return aprev == 4 and tau == 5
        else:
            raise Exception(f"Property {self.prop} not defined.")


    def find_regions(self, *args, **kwargs):
        """Returns lists of counterexample points and non-counterexample points.
        Overall, this is accomplished by randomly sampling regions, applying
        SyReNN, and seeing if they have a counterexample.
        """
        np.random.default_rng(self.seed)
        n_samples = kwargs["n_samples"]
        cexs = []
        points = []
        search_start = timer()
        while len(cexs) < n_samples:
            kwargs["n_samples"] = 100

            # Compute planes by sampling the input space.
            # regions is a list of planes, each defined by a list of 4 vertices.
            regions = self.compute_regions(*args, **kwargs)

            # syrenn is a list of UPolytopes. Each UPolytope is a list of Numpy arrays.
            # Each UPolytope is defined by vertices representing a partition of the input 
            # region corresponding to a linear region of the output.
            syrenn = self.network.transform_planes(
                regions, compute_preimages=True, include_post=False)
            cex_found = False
            for _, upolytope in enumerate(syrenn):
                all_points = np.concatenate(upolytope)
                results = self.property(all_points)
                if False in results:
                    cex_found = True
                    break
            if cex_found:
                property_false, property_true = self.find_sample_points(syrenn)
                cexs.extend(property_false)
                points.extend(property_true)
                print("Counterexamples found so far: ", len(cexs))
            if (timer() - search_start) > self.timeout:
                print("Timeout...")
                return None, None

        return cexs, points

    def find_sample_points(self, syrenn):
        """Given SyReNN for a region, returns points that violate or satisfy the property."""
        np.random.default_rng(self.seed)
        property_false = []
        property_true = []

        # Loop through list of arrays, which are a set of vertices defining a region.
        for upolytope in syrenn:
            # all_polys is all vertices for all regions.
            all_polys = np.concatenate(upolytope)
            min_ = np.min(all_polys, axis=0)
            max_ = np.max(all_polys, axis=0)
            all_polys = []
            
            # Loop through each region, defined by the array of vertices. 
            for pre_poly in upolytope:
                center = np.mean(pre_poly, axis=0)
                for alpha in [0.25, 0.5, 0.75]:
                    # Sample along the lines between each vertex and the center.
                    poly = pre_poly + (alpha * (pre_poly - center))
                    all_polys.extend(poly)
            results = self.property(all_polys)
            cex_points = np.asarray(all_polys)[results == False]
            property_false.extend(cex_points)

            sample_points = np.random.uniform(min_, max_, size=(5*len(cex_points), 5))
            results = self.property(sample_points)
            sample_points = sample_points[results][:len(cex_points)]
            property_true.extend(sample_points)

        return np.asarray(property_false), np.asarray(property_true)


    def property(self, inputs):
        """Returns array of booleans, True if the input satisfies the property, else False."""
        outputs = self.network.compute(inputs)
        labels = np.argmax(outputs, axis=1)
        if self.prop == 1:
            return np.asarray([o[0] >= -1500 for o in outputs])
        elif self.prop == 2:
            mins = np.argmin(outputs, axis=1)
            return mins != 0
        elif self.prop == 3 or self.prop == 4:
            return labels != 0
        elif self.prop == 5:
            return labels == 4
        elif self.prop == 6 or self.prop == 10:
            return labels == 0
        elif self.prop == 7:
            return np.asarray([labels[i]!=3 and labels[i]!=4 for i in range(len(labels))], dtype=bool)
        elif self.prop == 8:
            return np.asarray([labels[i]==0 or labels[i]==1 for i in range(len(labels))], dtype=bool)
        elif self.prop == 9:
            return labels == 3
        else:
            raise Exception(f"Property {self.prop} not defined.")
            

    def compute_regions(self,
                        intruder_heading, own_velocity, intruder_velocity,
                        rho, theta, process, n_samples):
        """Samples @n_samples 2D slices from the space."""
        regions = []
        for _ in range(n_samples):
            # Uniformly sample from 3 dimensions.
            self.intruder_heading = np.random.uniform(*intruder_heading)
            self.own_velocity = np.random.uniform(*own_velocity)
            self.intruder_velocity = np.random.uniform(*intruder_velocity)
            
            # Each point (rho[i], theta[j]) is a vertex of a plane 
            regions.append(process(np.array([
                self.build_input(rho[0], theta[0]),
                self.build_input(rho[1], theta[0]),
                self.build_input(rho[1], theta[1]),
                self.build_input(rho[0], theta[1]),
            ])))
        return regions

    def build_input(self, distance, theta):
        """Returns an (un-processed) input point corresponding to the scenario.
        """
        return np.array([distance, theta, self.intruder_heading,
                         self.own_velocity, self.intruder_velocity])

    def analyze(self):
        return

    def set_args(self, args):
        if len(args)==0:
            self.net = input("Which network? (e.g., acas_2_9): ")
            self.prop = int(input("Which property? [1..10]: "))
            self.num_points = int(input("How many points to generate? "))
            self.path = input("Path to save files: ")
        else:
            self.net = args[0]
            self.prop = int(args[1])
            self.num_points = int(args[2])
            self.path = args[3]
        assert(self.valid_network())
        assert(self.prop in [*range(1, 11)])


if __name__ == "__main__":
    args = [*sys.argv[1:]]
    exp = ACASPoints("acas_points")
    exp.set_args(args)
    exp.main()
