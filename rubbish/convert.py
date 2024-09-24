import torch


A = torch.load("./Aacc0.8936.pth")
B = torch.load("./Bacc0.8939.pth")


A_1B = {}
B_1A = {}

for (kA, vA), (kB, vB) in zip(A.items(), B.items()):
    if "blocks.0" in kA:
        vA, vB = vB, vA
    A_1B[kA] = vA
    B_1A[kB] = vB

torch.save(A_1B, "A_1B.pth")
torch.save(B_1A, "B_1A.pth")


A_2B = {}
B_2A = {}

for (kA, vA), (kB, vB) in zip(A.items(), B.items()):
    if ("blocks.0" in kA) or ("blocks.1" in kA):
        vA, vB = vB, vA
    A_2B[kA] = vA
    B_2A[kB] = vB

torch.save(A_2B, "A_2B.pth")
torch.save(B_2A, "B_2A.pth")


A_bB = {}
B_bA = {}

for (kA, vA), (kB, vB) in zip(A.items(), B.items()):
    if ("blocks.1" in kA) or ("blocks.3" in kA):
        vA, vB = vB, vA
    A_bB[kA] = vA
    B_bA[kB] = vB

torch.save(A_bB, "A_bB.pth")
torch.save(B_bA, "B_bA.pth")