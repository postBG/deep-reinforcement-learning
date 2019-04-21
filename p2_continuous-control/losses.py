import torch


def calculate_clipped_surrogate(advantages, old_log_probs, new_log_probs, epsilon):
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    clipped_surrogate = torch.min(surr, clipped)

    return torch.mean(clipped_surrogate)
