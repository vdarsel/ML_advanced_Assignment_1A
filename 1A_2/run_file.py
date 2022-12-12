import numpy as np
from Tree import Tree
from Tree import Node
from ipywidgets import interact

def main():
    file=["q1A_2_small_tree.pkl","q2_2_medium_tree.pkl","q2_2_large_tree.pkl"]
    for a in (file):
        run(a)

def run(file):
    print("\n1. Load tree data from file and print it\n")

    filename = "data/test_data/"+file
    print("filename: ", filename)

    t = Tree()
    t.load_tree(filename)
    t.print()
    print("K of the tree: ", t.k, "\talphabet: ", np.arange(t.k))

    print("\n2. Calculate likelihood of each FILTERED sample\n")
    # These filtered samples already available in the tree object.
    # Alternatively, if you want, you can load them from corresponding .txt or .npy files

    for sample_idx in range(t.num_samples):
        beta = t.filtered_samples[sample_idx]
        print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        sample_likelihood = t.Likelihood(beta)
        print("\tLikelihood: ", sample_likelihood)


if __name__ == "__main__":
    main()
