import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import mlflow

def main():
    """Main function of the script."""
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))
    print("input data:", args.data)
    file_name = os.path.join(args.data)
    titanic_df = pd.read_csv(file_name)
    print(titanic_df.head(10))
    
    mlflow.log_metric("num_samples", titanic_df.shape[0])
    mlflow.log_metric("num_features", titanic_df.shape[1] - 1)
    
    titanic_train_df, titanic_test_df = train_test_split(
        titanic_df,
        test_size=args.test_train_ratio,
    )
    # output paths are mounted as folder, therefore, we are adding a filename to the path
    titanic_train_df.to_csv(os.path.join(args.train_data, "data.csv"), index=True)
    titanic_test_df.to_csv(os.path.join(args.test_data, "data.csv"), index=True)
    # Stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()
