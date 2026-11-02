"""Run PyTorch scripts that assume CUDA on another device, with every redirected call on record.

Nothing here runs unless ``cumlx run`` starts a script. Importing ``cumlx`` changes nothing in torch.
"""
