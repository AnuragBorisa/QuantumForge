# Day 8: GAN Theory

## 1. What is a GAN?
- Two-player game between Generator and Discriminator.
- Objective: G learns to mimic data distribution; D learns to tell real vs fake.

## 2. Roles
| Network        | Input | Output        | Goal                     |
| -------------- | ----- | ------------- | ------------------------ |
| Generator (G)  | noise z | fake sample | D(fake) → 1 (fool D)     |
| Discriminator (D) | sample x | prob [0–1] | classify real(1) vs fake(0) |

## 3. Adversarial Loss
- **Discriminator:**  
  \`L_D = -[ Eₓ log D(x) + E_z log (1–D(G(z))) ]\`
- **Generator:**  
  \`L_G = - E_z log D(G(z))\`

## 4. Training Procedure
1. Update D on real & fake batches  
2. Update G to maximize D’s belief in fakes  
3. Alternate for N epochs

## 5. Training Loop (pseudocode)
```python
for epoch in range(num_epochs):                                  
    # ───────────────────────────────────────────────────────────
    # “Each epoch” = one full pass over all our homework (the dataset)
    print(f"Epoch {epoch+1}/{num_epochs} begins")

    for real_batch in data_loader:                              
        # “Each batch” = teacher sees a small stack of real examples
        #              = student makes a small batch of guesses

        # 1) Train the Teacher (Discriminator D)
        # ────────────────────────────────────────────────────────

        # a) Student (G) makes guesses from random noise
        z = sample_noise(batch_size, dim_z)                     
        # “Give G some random hints → G turns them into fake samples”
        
        fake_batch = G(z).detach()                             
        # “Teacher grades these fakes but we ‘detach’ so G’s brain doesn’t get changed now”
        
        # b) Teacher scores real vs. fake
        pred_real = D(real_batch)       # score for real: D says “this is real x%”
        pred_fake = D(fake_batch)       # score for fake: D says “this is real y%”
        
        # c) Compute how badly the teacher did
        #    - For real data, we want D(real) close to 1 (100% real)
        #    - For fake data, we want D(fake) close to 0 (0% real)
        loss_D = BCE(pred_real, 1) + BCE(pred_fake, 0)         
        # “Total penalty: sum of ‘teacher was surprised’ on real + fake”
        
        # d) Clear old error notes so teacher can write new ones
        optimD.zero_grad()                                     
        # “erase last round’s margin notes”
        
        # e) Teacher writes new margin notes on how to improve
        loss_D.backward()                                      
        # “calculate exactly which rubric rules to tweak (gradients)”
        
        # f) Teacher updates its checking strategy
        optimD.step()                                          
        # “apply the margin notes to rewrite the rubric”
        # “With the new rubric in place, the teacher is now better
        #  at spotting which guesses are truly good or bad.”

        # 2) Train the Student (Generator G)
        # ────────────────────────────────────────────────────────

        # a) Student makes fresh guesses
        z2 = sample_noise(batch_size, dim_z)                   
        fake_batch2 = G(z2)                                   
        # “new random hints → new fake samples”

        # b) Teacher scores the new guesses
        pred_fake2 = D(fake_batch2)                           
        # “teacher grades the latest essays”

        # c) Compute student’s penalty: student wants teacher to say “100% real”
        loss_G = BCE(pred_fake2, 1)                           
        # “penalty = how surprised teacher is that fake is real”
        
        # d) Clear student’s old error notes
        optimG.zero_grad()                                     
        # “erase last round’s study notes”
        
        # e) Student writes new study plan on how to guess better
        loss_G.backward()                                      
        # “calculate which guess-direction gives bigger teacher score”
        
        # f) Student updates its guessing strategy
        optimG.step()                                          
        # “apply the study plan so next guesses get higher scores”

    # ───────────────────────────────────────────────────────────
    # Optional: Show how far both have come this epoch
    print(f"End of epoch {epoch+1}: teacher loss={loss_D.item():.3f}, student loss={loss_G.item():.3f}")
```

## 6. Notes & Observations
- e.g. “If D gets too strong, G’s gradients vanish; use label smoothing.”  
- e.g. “Tried 5 D steps per G step → more stable early on.”

