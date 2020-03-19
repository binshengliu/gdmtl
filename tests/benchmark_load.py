from timeit import default_timer

start = default_timer()
path = (
    "/research/remote/petabyte/users/binsheng/"
    "sum-ranker/data/datasets/msmarco/collection.tsv"
)
with open(path, "r") as f:
    lines = f.readlines()
print(f"{default_timer() - start}")

start = default_timer()
d1 = {}
for line in lines:
    splits = line.split(maxsplit=1)
    d1[int(splits[0])] = splits[1]
print(f"{default_timer() - start}")

start = default_timer()
d2 = dict.fromkeys(range(8841823))
for line in lines:
    splits = line.split(maxsplit=1)
    d2[int(splits[0])] = splits[1]
print(f"{default_timer() - start}")
