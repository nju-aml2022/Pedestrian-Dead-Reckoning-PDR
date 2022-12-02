import os
import argparse
from pdr import pdr
from dataloader import TestCase
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import sys
from itertools import product

DATA_DIR_ROOT = "Dataset-of-Pedestrian-Dead-Reckoning/"
WALK=["Hand-Walk", "Bag-Walk", "Pocket-Walk"]
EXCLUDE=["Hand-Walk/Hand-Walk-05-001", "Hand-Walk/Hand-Walk-05-002"]

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
    datasets = [dataset for dataset in datasets if dataset not in EXCLUDE]
    
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
            print(f"Error when processing dataset {dataset}: {e}")
            # traceback.print_exc()
            fails+=1
            continue
    df = pd.DataFrame({"name": names, "dist_error": dist_error, "dir_error": dir_error, "dir_ratio": dir_ratio})
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

def worker(_args):
    try:
        test_case = TestCase(os.path.join(DATA_DIR_ROOT, _args["dataset"]))
        pdr(
            test_case, 
            model_name=_args["model_name"],
            distance_frac_step=_args["distance_frac_step"],
            clean_data=_args["clean_data"], 
            optimized_mode_ratio=_args["optimized_mode_ratio"],
            butter_Wn=_args["butter_Wn"])
        dist_error = test_case.get_dist_error()
        dir_error = test_case.get_dir_error()
        dir_ratio = test_case.get_dir_ratio()
        return _args["dataset"], _args["model_name"], _args["distance_frac_step"], _args["clean_data"], _args["optimized_mode_ratio"], _args["butter_Wn"], dist_error, dir_error, dir_ratio
    except Exception as e:
        return _args["dataset"], _args["model_name"], _args["distance_frac_step"], _args["clean_data"], _args["optimized_mode_ratio"], _args["butter_Wn"], 0, 0, 0

def mute():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


def mp_test():
    datasets = []
    for dir in WALK:
        datasets += [os.path.join(dir, name) for name in os.listdir(os.path.join(DATA_DIR_ROOT, dir))]
    datasets = [dataset for dataset in datasets if dataset not in EXCLUDE]
    args = {
        "model_name": {
            "default":"RandomForest",
            "options": ["SVR", "RandomForest", "Mean", "AdaBoost", "ExtraTree"]
        },
        "distance_frac_step": {
            "default": 5,
            "options": [3,4,5,6,7]
        },
        "clean_data": {
            "default": 5,
            "options": [3,4,5,6,7,8,9,10]
        },
        "optimized_mode_ratio": {
            "default": 0.9,
            "options": [0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0]
        },
        "butter_Wn": {
            "default": 0.005,
            "options": [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
        },
    }


    datasets.append("../test_case0")
    todos=[]
    # for dataset in datasets:
    #     for key, value in args.items():
    #         for option in value["options"]:
    #             _args = {
    #                 "dataset": dataset,
    #                 "model_name": args["model_name"]["default"],
    #                 "distance_frac_step": args["distance_frac_step"]["default"],
    #                 "clean_data": args["clean_data"]["default"],
    #                 "optimized_mode_ratio": args["optimized_mode_ratio"]["default"],
    #                 "butter_Wn": args["butter_Wn"]["default"]
    #             }
    #             _args[key] = option
    #             todos.append(_args)

    for todo in product(datasets,
        args["model_name"]["options"], 
        args["distance_frac_step"]["options"],
        args["clean_data"]["options"],
        args["optimized_mode_ratio"]["options"],
        args["butter_Wn"]["options"]):
        todos.append({
            "dataset": todo[0],
            "model_name": todo[1],
            "distance_frac_step": todo[2],
            "clean_data": todo[3],
            "optimized_mode_ratio": todo[4],
            "butter_Wn": todo[5]
        })
    


    with mp.Pool(processes=72, initializer=mute) as pool:
        res = list(tqdm(pool.imap(worker, todos), total=len(todos)))
    df = pd.DataFrame(res, columns=["dataset", "model_name", "distance_frac_step", "clean_data", "optimized_mode_ratio", "butter_Wn", "dist_error", "dir_error", "dir_ratio"])
    df.to_csv("result_mp_adjust_case0.csv", index=False)
        

if __name__ == "__main__":
    mp_test()
