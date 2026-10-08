# Scaling-law report (simulated backend)

- experiments: 41 tracked, 41 completed, 0 failed
- budget charged: 9.69 B200-hours of 12

## Throughput model

- seconds/token = 1.343e-06 + 8.081e-16 * flops/token (asymptotic MFU 55.0%)
- fit on 41 runs, log-space R^2 = 0.951, RMSE of log(tokens/s) = 0.088

## Learning-rate rule

- lr_opt = 0.07664 * N^-0.2046
  - k=6 (N=10.6M): lr_opt = 2.80e-03, losses [4.5219, 4.4505, 4.4294, 4.4252, 4.4554] at lrs ['4.9e-04', '9.9e-04', '2.0e-03', '4.0e-03', '7.9e-03']
  - k=10 (N=50.8M): lr_opt = 2.03e-03, losses [4.0898, 4.0365, 4.0139, 4.0007, 4.0395] at lrs ['3.6e-04', '7.2e-04', '1.4e-03', '2.9e-03', '5.8e-03']

## Iso-time profiles

| budget | best k | N_opt (grid) | D_opt (grid) | D/N | loss | runtime | N_opt (interp.) | loss (interp.) |
|---|---|---|---|---|---|---|---|---|
| 5.0min | 16 | 205.6M | 0.12B | 0.6 | 3.8660 | 290s | 94.2M | 3.8852 |
| 10.0min | 16 | 205.6M | 0.24B | 1.1 | 3.5537 | 581s | 136.5M | 3.5598 |
| 20.0min | 16 | 205.6M | 0.47B | 2.3 | 3.3079 | 1163s | 209.4M | 3.3079 |
| 40.0min | 16 | 205.6M | 0.95B | 4.6 | 3.1298 | 2332s | 272.8M | 3.1296 |
| 1.33h | 16 | 205.6M | 1.90B | 9.3 | 2.9843 | 4672s | 279.0M | 2.9740 |

Power laws below are fit to the parabola-interpolated minima (Hoffmann et al. approach 2).
- N_opt = 9.975e+06 * T^0.4121 (log-space R^2 = 0.930)
- D_opt = 1.237e+06 * T^0.8430 (log-space R^2 = 0.998)
- L_opt = 2.5019 + 12.24 * T^-0.3846 (R^2 = 1.000)

## Parametric loss surface

- L(N, D) = 2.2017 + 391.6 / N^0.3662 + 2118 / D^0.3965
- fit on 16 runs: RMSE = 0.0108, max |error| = 0.0181, R^2 = 0.9996
- implied FLOPs-optimal exponents: N_opt ~ C^0.520, D_opt ~ C^0.480

## Hold-out validation (largest time budget held out)

- held out T = 1.33h (13 training runs)
- parametric: RMSE = 0.0349, max |error| = 0.0453; predicted best k = 20 vs actual best k = 16
- iso-time N_opt law: predicted 322.6M vs actual 279.0M
- iso-time L_opt law: predicted 2.9694 vs actual 2.9740

## Final run (48.00h budget, tokens sized for 85% of it)

- **parametric**: k=31 -> 31L x 1984d (ff 5376, 31 heads) N_ne=1480.2M total=1607.1M, D=16.49B tokens (D/N=11.1), batch 1024 x 512, peak lr 1.02e-03, predicted runtime 40.80h, predicted loss 2.5620
- **iso-time**: k=30 -> 30L x 1920d (ff 5120, 30 heads) N_ne=1327.2M total=1450.1M, D=18.02B tokens (D/N=13.6), batch 1024 x 512, peak lr 1.04e-03, predicted runtime 40.79h, predicted loss 2.6279
- FLOPs-based Chinchilla optimum at C=1.543e+20: N=1.21B, D=21.18B, loss 2.5571
- bootstrap over runs (20 refits): predicted loss at the chosen config median 2.5695, 10-90% [2.4953, 2.6370]; optimal shape across refits: k=24: 1, k=29: 3, k=30: 2, k=31: 2, k=32: 2, k=33: 2, k=34: 3, k=35: 1, k=38: 1, k=39: 3

**Chosen (parametric)**: predicted final loss **2.5695** (bootstrap median)

```json
{
  "architecture_config": {
    "attention_bias": false,
    "head_dim": 64,
    "hidden_size": 1984,
    "intermediate_size": 5376,
    "num_attention_heads": 31,
    "num_hidden_layers": 31,
    "num_key_value_heads": 31,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000,
    "tie_word_embeddings": false,
    "dtype": "bfloat16",
    "vocab_size": 32000
  },
  "optimizer_config": {
    "lr_scheduler": {
      "peak_value": 0.00102,
      "final_lr_frac": 0.1,
      "warmup_frac": 0.05,
      "init_value": 0.0
    },
    "weight_decay": 0.01,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-08,
    "eps_root": 1e-08,
    "grad_clip_norm": 1.0
  },
  "train_batch_size": 1024,
  "val_batch_size": 32,
  "n_evals": 32,
  "total_train_tokens": 16492003328,
  "max_runtime_seconds": 172800.0,
  "model_seed": 0
}
```
