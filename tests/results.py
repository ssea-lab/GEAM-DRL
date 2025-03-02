import re


def main():
    # Extract only numbers
    nums = re.findall(r"\d+\.\d+", s)
    # Print them out
    for i in range(0, len(nums), 6):
        for j in range(6):
            print(nums[i + j], end="")
            if j < 5:
                print("\t", end="")
        print()


if __name__ == "__main__":
    # Read 's' from aaa.txt
    with open("aaa.txt", "r") as f:
        s = f.read()
    main()
