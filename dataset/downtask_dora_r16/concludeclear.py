import os
import re
import sys
path = sys.argv[1]




def find_last_accuracy_value(file_pointer):
    original_position = file_pointer.tell()
    file_pointer.seek(0, 2)
    end_position = file_pointer.tell()
    buffer_size = 1024
    while end_position > 0:
        start_position = max(0, end_position - buffer_size)
        file_pointer.seek(start_position)
        chunk = file_pointer.read(end_position - start_position)
        lines = chunk.splitlines(True)  # True keeps the newline character with the line
        if start_position > 0:
            first_line_in_chunk = lines[0]
            lines[0] = file_pointer.readline() + first_line_in_chunk
        for line in reversed(lines):
            match = re.search(r"accuracy\s\d+\s+(\d+\.\d+)", line)
            if match:
                accuracy_value = float(match.group(1)) * 100
                file_pointer.seek(original_position)
                return accuracy_value
        end_position = start_position
    file_pointer.seek(original_position)
    return None




print("\n\n\n\n\n======================================================")
files = os.listdir(path)
result = {}
for file in files:
    file = os.path.join(path, file)
    if ".txt" in file:
        name = os.path.basename(file).split(".")[0]
        with open(file, "r") as f:
            value = find_last_accuracy_value(f)
        result[name] = value * 100.0
        print(f"{name}: {value:.2f}")
        os.remove(file)
with open("last_result.txt", "w") as f:
    for k, v in result.items():
        f.write(f"{k}: {v:.2f}\n")
print("\n\n")
