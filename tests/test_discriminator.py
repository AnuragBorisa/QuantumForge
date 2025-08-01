import torch
import torch.nn.functional as F
from qfinqgan.discriminator import ClassicalDiscriminator

def test_predict_proba_shape_and_range():
    batch_size = 7

    # 1) create a random batch of one-hot vectors
    labels   = torch.randint(0, 8, (batch_size,))
    inputs   = F.one_hot(labels, num_classes=8).float()  # shape (7,8)

    # 2) instantiate & call predict_proba
    model    = ClassicalDiscriminator()
    probs    = model.predict_proba(inputs)

    # 3) assertions
    assert isinstance(probs, torch.Tensor)
    assert probs.shape == (batch_size, 1)
    # all values must lie in [0,1]
    assert torch.all(probs >= 0) and torch.all(probs <= 1)
