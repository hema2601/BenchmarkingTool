import src.benchmark as b
import sys

argc = len(sys.argv)

if argc < 2:
    print("Usage: python3 benchmark.py <Path to TOML-config>")

filename = sys.argv[1]
bo = b.BenchmarkObj.from_config(b.ConfigObj(sys.argv[2]))

bo.run_benchmark()

