# Gradient Descent Alphabet Recognition

A comprehensive test of gradient-based optimization on a simple letter classification task.

## The Challenge

Classify letters A-Z from 100x100 pixel images using a small neural network.

```
Architecture: 10,000 inputs → 32 hidden → 26 outputs
Parameters: ~321,000
Activations tested: ReLU and Tanh
```

Augmentation applied during training:
- Rotation: ±25 degrees
- Position jitter: ±20%
- Font size variation: 12pt and 64pt
- Color inversion: white-on-black and black-on-white

Training/Validation split:
- 50 variations per letter for training (1,300 total)
- 10 variations per letter for validation (260 total)
- Different random seeds ensure validation images are never seen during training

## Results

Tested 6 optimizers × 2 activation functions = 12 experiments.

### ReLU Activation

| Optimizer | Best Accuracy | Random Baseline | Improvement |
|-----------|---------------|-----------------|-------------|
| Adam | 6.54% | 3.85% | +2.69% |
| AdamW | 5.77% | 3.85% | +1.92% |
| Adagrad | 5.77% | 3.85% | +1.92% |
| Adadelta | 5.77% | 3.85% | +1.92% |
| RMSprop | 5.00% | 3.85% | +1.15% |
| SGD | 4.62% | 3.85% | +0.77% |

### Tanh Activation

| Optimizer | Best Accuracy | Random Baseline | Improvement |
|-----------|---------------|-----------------|-------------|
| Adagrad | 6.15% | 3.85% | +2.30% |
| Adam | 5.38% | 3.85% | +1.53% |
| AdamW | 5.38% | 3.85% | +1.53% |
| RMSprop | 5.38% | 3.85% | +1.53% |
| Adadelta | 5.38% | 3.85% | +1.53% |
| SGD | 5.00% | 3.85% | +1.15% |

**Best overall: Adam + ReLU at 6.54%**

**Best with tanh (matching evolutionary model): Adagrad at 6.15%**

Congratulations to Adam for being 2.69% better than a coin flip.

## For Comparison

An evolutionary approach using the exact same architecture (with tanh) achieves 78%+ accuracy and continues to improve.

Same network. Same task. Same augmentation. Same everything except the training method.

The best gradient result with tanh: 6.15%
The evolutionary result with tanh: 78%+

And that's on a single font.

I have another evolutionary model training on 5 different fonts simultaneously. It's currently in the 40%+ range and climbing. With 64 hidden neurons instead of 32, learning font-invariant representations from scratch.

Meanwhile, the best gradient optimizer couldn't break 7% on one font with either activation function.

But I'm sure gradients can do this easily. Everyone keeps telling me so.

## Usage

```bash
# Run all optimizers
python alphabet_gradient_descent.py --fonts 1

# Results saved to optimizer_comparison_<timestamp>/
```

## Requirements

```bash
pip install torch numpy pillow
```

## FAQ

**Q: Did you tune the hyperparameters?**

A: The code is right there. Tune them yourself. Let me know when you break 10%.

**Q: What about deeper networks?**

A: The challenge is 32 hidden neurons. That's the constraint. If you need more capacity to solve a 26-class letter classification task, that says something.

**Q: What about batch normalization / dropout / skip connections?**

A: Feel free to add them. The architecture is intentionally minimal to match the evolutionary baseline. Fork the repo and show me your results.

**Q: This isn't a fair comparison.**

A: It's the same architecture, same data, same augmentation. What would make it fair?

**Q: Gradient descent wasn't designed for this.**

A: Gradient descent wasn't designed for a 26-class classification task with data augmentation? Interesting.

## The Point

This repo exists because people kept telling me that gradient descent could easily solve any task my evolutionary models solve.

Here's the code. Here are six optimizers. Here are the logs.

Show me.

## License

MIT

Do whatever you want with it. Add a VAE, CNN, transformer, whatever you want. It can't do what mine can on the same playing field.

Hop off your backprop dildos and realize there's a new player in the game.
