# Concise outreach package for Umangi Jain

## Suggested subject

Reproducing Material Magic Wand + an active graph-correction experiment

## Draft message

Hi Umangi,

I am an MSc student at U of T interested in TISL's work on interactive 3D perception. I read Material Magic Wand and GaussianCut, then built a reproduction-first experiment that keeps the released Material Magic Wand encoder frozen and tests calibrated graph inference with explicit positive/negative clarification.

I used the released 100-mesh/241-query benchmark, verified every render archive and embedding cache, froze all parameters on a mesh-disjoint validation split, and evaluated the remaining 94 meshes. The honest result is: frozen graph inference does not improve mean held-out F1 (0.4794 versus 0.7368). After three simulated binary corrections, held-out mean F1 is 0.6222 versus 0.7368 for the Material Magic Wand threshold baseline; the paired mesh-bootstrap delta is -0.1146 [-0.1812, -0.0551].

I also built an interactive demo, failure analysis, and fully locked evaluation. I am not claiming a new state of the art - I would especially value your advice on whether the preprocessing matches your intended release pipeline and whether negative clarification is a useful direction for the project.

Would you be open to a short conversation? I can send a concise technical summary, exact reproduction commands, and the demo.

Best,  
[Name]

## Attach or link

1. One-page technical summary (`docs/technical_summary.md`).
2. Method figure and principal result table.
3. Short demo GIF and three-minute script.
4. Repository only after the numerical audit and clean-environment reproduction pass.

## Conversation points

- Confirm released preprocessing and the intended threshold sign convention.
- Ask whether the original 5-mesh/13-query validation identities can be shared.
- Discuss whether negative evidence matches likely artist workflows.
- Propose a small real-user click study rather than over-interpreting simulated feedback.
