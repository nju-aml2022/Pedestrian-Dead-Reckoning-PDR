import argparse
from pdr import pdr
from dataloader import TestCase
import time

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="dataset location", required=True)
parser.add_argument("--silent", action="store_true", help="whether to print distance and direction error")

args = parser.parse_args()
start=time.time()
test_case = TestCase(args.dataset)
pdr(test_case)
if not args.silent:
    print(f"Done in {time.time()-start} seconds")
    if test_case.have_location_valid:
        dist_error = test_case.get_dist_error()
        dir_error = test_case.get_dir_error()
        dir_ratio = test_case.get_dir_ratio()
        print("Distances error: ", dist_error)
        print("Direction error: ", dir_error)
        print("Direction ratio: ", dir_ratio)
    else:
        print("Location.csv not found, cannot calculate distance and direction error")
