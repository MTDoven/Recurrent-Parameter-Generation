import os
checkpoint_folder = "./checkpoint"
test_command = "CUDA_VISIBLE_DEVICES=3 python test.py {}"

checkpoint_list = os.listdir(checkpoint_folder)
for file in checkpoint_list:
    file = os.path.join(checkpoint_folder, file)
    print(f"start testing: {file}")
    os.system(test_command.format(file))
    print("\n============================================================================\n\n\n")