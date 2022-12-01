import os
import argparse
from pdr import pdr
from dataloader import TestCase
import pandas as pd
from tqdm import tqdm
import traceback

DATA_DIR_ROOT = "../Dataset-of-Pedestrian-Dead-Reckoning/"
WALK=["Hand-Walk", "Bag-Walk", "Pocket-Walk"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use", type=str, default="walk", choices=["walk", "other", "all"], help="datasets to use")
    parser.add_argument("--method", type=str, default="Mean", choices=["Mean", "DecisionTree", "Linear", "SVR", "RandomForest", "AdaBoost", "GradientBoosting", "Bagging", "ExtraTree"], help="step predictor method")
    args = parser.parse_args()
    datasets = []
    if args.use == "walk":
        for dir in WALK:
            # get names of all directories in `dir`
            datasets += [os.path.join(dir, name) for name in os.listdir(os.path.join(DATA_DIR_ROOT, dir))]
    elif args.use == "other":
        for dir in os.listdir(DATA_DIR_ROOT):
            if dir not in WALK:
                datasets += [os.path.join(dir, name) for name in os.listdir(os.path.join(DATA_DIR_ROOT, dir))]
    else:
        for dir in os.listdir(DATA_DIR_ROOT):
            datasets += [os.path.join(dir, name) for name in os.listdir(os.path.join(DATA_DIR_ROOT, dir))]

    names=[]
    dist_error=[]
    dir_error=[]
    dir_ratio=[]
    fails=0
    for dataset in tqdm(datasets ,desc="Testing"):
        try:
            test_case = TestCase(os.path.join(DATA_DIR_ROOT, dataset))
            pdr(test_case, optimized_mode_ratio=0.9, butter_Wn=0.005, model_name=args.method)
            dist_error.append(test_case.get_dist_error())
            dir_error.append(test_case.get_dir_error())
            dir_ratio.append(test_case.get_dir_ratio())
            names.append(dataset)
        except Exception as e:
            # print(f"Error when processing dataset {dataset}:")
            # traceback.print_exc()
            fails+=1
            continue
    df = pd.DataFrame({"dataset": names, "dist_error": dist_error, "dir_error": dir_error, "dir_ratio": dir_ratio})
    df.to_csv("result.csv", index=False)
    print("Done, results saved to result.csv")
    print(f"Average distance error: {df['dist_error'].mean()}")
    print(f"Average direction error: {df['dir_error'].mean()}")
    print(f"Average direction ratio: {df['dir_ratio'].mean()}")
    print(f"Failed count: {fails}")

def test_step_method():
    METHODS=['Mean','DecisionTree', 'Linear', 'SVR', 'RandomForest', 'AdaBoost', 'GradientBoosting', 'Bagging', 'ExtraTree']
    datasets=[]
    for dir in WALK:
        # get names of all directories in `dir`
        datasets += [os.path.join(dir, name) for name in os.listdir(os.path.join(DATA_DIR_ROOT, dir))]

    dist_error=[]
    dir_error=[]
    dir_ratio=[]
    success=[]
    for method in METHODS:
        _dist_error=[]
        _dir_error=[]
        _dir_ratio=[]
        for dataset in tqdm(datasets ,desc=method):
            try:
                test_case = TestCase(os.path.join(DATA_DIR_ROOT, dataset))
                pdr(test_case, optimized_mode_ratio=0.9, butter_Wn=0.005, model_name=method)
                _dist_error.append(test_case.get_dist_error())
                _dir_error.append(test_case.get_dir_error())
                _dir_ratio.append(test_case.get_dir_ratio())
            except:
                continue
        dist_error.append(sum(_dist_error)/len(_dist_error))
        dir_error.append(sum(_dir_error)/len(_dir_error))
        dir_ratio.append(sum(_dir_ratio)/len(_dir_ratio))
        success.append(len(_dist_error))
    df = pd.DataFrame({"method": METHODS, "dist_error": dist_error, "dir_error": dir_error, "dir_ratio": dir_ratio, "success": success})
    df.to_csv("result_step_method.csv", index=False)
        

if __name__ == "__main__":
    main()
