import torch
import torch.nn.functional as F

def se_loss_batched(A, partitions, max_classes):
    assert A.dim() == 3 and partitions.dim() == 2
    assert A.shape[0] == partitions.shape[0]  # batch size
    assert A.shape[1] == A.shape[2] == partitions.shape[1]  # num_nodes
    device = A.device
    batch_size, num_nodes = partitions.shape
    num_classes = max_classes
    partitions = F.one_hot(partitions, num_classes=num_classes).to(
        dtype=A.dtype, device=device)

    # Remove self-loops from A
    eye = torch.eye(A.size(1), device=A.device).unsqueeze(0)
    A = A - eye * A.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)

    # Compute Deno_sumA, enco_p and encolen in a batched manner
    sumA = A.sum(dim=(-2, -1), keepdim=True)
    Deno_sumA = 1 / sumA

    # Perform batched matrix multiplications
    C = partitions.float()  # shape [batch_size, num_nodes, num_classes]
    Rate_p = torch.matmul(C.transpose(1, 2), torch.matmul(A, C)) * Deno_sumA
    # Calculate enco_p and encolen
    IsumC = torch.ones(batch_size, 1, num_nodes, device=A.device)
    IsumCDC = torch.ones(batch_size, num_classes, 1, device=A.device)
    enco_p = torch.matmul(IsumCDC, torch.matmul(
        IsumC, torch.matmul(A, C))) * Deno_sumA
    encolen = torch.log2(enco_p + 1e-20)

    # Sum over the batch dimension
    se_loss = torch.sum(torch.einsum('bii->b', Rate_p.mul(encolen)))

    return se_loss
