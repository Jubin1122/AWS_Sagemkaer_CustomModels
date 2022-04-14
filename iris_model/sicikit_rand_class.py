from sklearn.ensemble import RandomForestClassifier
# from __future__ import print_function

import argparse
import os

import joblib
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
#     parser.add_arguments("--max_depth", type=int,default=2)
#     parser.add_argument("--n_estimator", type=int)
#     parser.add_argument("--oob_score", action='store_true')
#     parser.add_argument("--n_jobs", type=int, default=-1)
#     parser.add_argument("--random_state", type=int, default= 0)
#     parser.add_argument("--max_feature", type=str, default= 50)
#     parser.add_argument("--min_samples_leaf", type=int)
    

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
    if len(input_files) == 0:
        raise ValueError(
            (
                "There are no files in {}.\n"
                + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                + "the data specification in S3 was incorrectly specified or the role specified\n"
                + "does not have permission to access the data."
            ).format(args.train, "train")
        )
    raw_data = [pd.read_csv(file, header=None, engine="python") for file in input_files]
    train_data = pd.concat(raw_data)

    # labels are in the first column
    train_y = train_data.iloc[:, 0]
    train_X = train_data.iloc[:, 1:]

    # Here we support a single hyperparameter, 
#     max_depth=args.max_depth
#     n_estimator = args.n_estimator
#     oob_score = args.oob_score
#     n_jobs = args.n_jobs
#     random_state = args.random_state
#     max_feature = args.max_feature
#     min_samples_leaf = args.min_samples_leaf
    

    # Now use scikit-learn's decision tree classifier to train the model.
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf = clf.fit(train_X, train_y)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf