## ATEX_code
# Please run the src/experiment2.py to reproduce the experiment results.
- train.py `function train()`: Train baseline model
- advtrain_explain_tangval.py `function advTrain_explanation_indirect()`: Train ATEX model
- advtrain_explain_tangval.py `function advTrain_FGSM()`: Evaluate model adversairal defence performance
- attack_explain.py `function attack_explanation()` : Evaluate model robustness before and after defense
- compare_explain.py `function compare_explanation()` : Compare model attention maps with SmoothGrad before and after defense

- attack_explain.py `function attack_explanation_random()` : Visualization of Adversarial attack on interpretation
- attack_explain.py `function explanation_random()` : Visualization saliency map of ATEX model and Baseline model

# we have clear code annotations for you to understand the code.
