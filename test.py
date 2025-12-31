import os
import shelve
import sys
import numpy as np

# def main():
#     if len(sys.argv) > 1:
#         shelf_path = sys.argv[1]
#     else:
#         shelf_path = os.path.join("data/set_0", "measurements.db")

#     if not os.path.exists(shelf_path):
#         print(f"Missing shelve file: {shelf_path}")
#         print("Pass a path, e.g. python test.py data/set_1/measurements.db")
#         return

#     with shelve.open(shelf_path) as db:
#         for key in sorted(db.keys()):
#             print(f"{key}: {db[key]}")


# if __name__ == "__main__":
#     main()
