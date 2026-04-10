import os

def print_tree(root, max_depth=2, prefix=""):
    if max_depth < 0:
        return
    entries = sorted(os.listdir(root))
    for i, name in enumerate(entries):
        path = os.path.join(root, name)
        connector = "├─ " if i < len(entries)-1 else "└─ "
        print(prefix + connector + name)
        if os.path.isdir(path):
            print_tree(path, max_depth-1, prefix + "│  ")

if __name__ == "__main__":
    print_tree(os.getcwd(), max_depth=2)