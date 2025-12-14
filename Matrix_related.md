## 1. Common Size Notations You’ll See

<img width="587" height="516" alt="image" src="https://github.com/user-attachments/assets/69457dfb-4d78-44a1-bd0a-b2dd5d0fee49" />

## 2. Formula

f(x) = wᵀ x <br><br>
**Step 1: x = input data (features). x ∈ ℝᵈ**
```text
x =
[
  x₁
  x₂
  x₃
  ⋮
  x_d
]
```
Example: 
- height
- weight
- age

**Step 2: w = weights**
```text
w =
[ w₁
  w₂
  w₃
  ...
  w_d ]
```

**Step 3: wᵀ = transpose of w**
```text
## This is done so we can multiply.
wᵀ = [ w₁  w₂  w₃  ...  w_d ]
```

**Step 4: wᵀx = dot product**
```
## Multiply each feature by its weight, then add them up.
wᵀx = w₁x₁ + w₂x₂ + w₃x₃ + ... + w_dx_d
```

**Step 4: What does f(x) give ?**
```
# A single number
f : ℝᵈ → ℝ
```
This number is:
- prediction (regression)
- score (classification)
- logit (neural networks)
